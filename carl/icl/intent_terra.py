"""Isolated, bounded Codex annotation of preselected Sokoban pairs.

The worker sees only paired boards and the predicate vocabulary.  It does not
receive prior annotations, actions, offsets, source names, or outcome metrics.
Each process is a fresh Codex session; batches can still couple the pairs.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_data import EffectType
from carl.icl.intent_data import SokobanPredicate
from carl.icl.intent_data import validate_intent

PROMPT_VERSION = "sokoban-pair-effects-v1"


class AnnotatorModel(StrEnum):
    TERRA = "gpt-6.1-terra"
    LUNA = "gpt-6-luna"


class ReasoningEffort(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


# Preserve the original public default while allowing an explicit alternative.
MODEL = AnnotatorModel.TERRA


class AnnotationStatus(StrEnum):
    ACCEPTED = "accepted"
    LOCAL_FALLBACK_REQUIRED = "failed_local_fallback_required"


class WorkerPhase(StrEnum):
    STARTED = "started"
    FINISHED = "finished"


class UsageStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_REPORTED = "not_reported"
    AMBIGUOUS = "ambiguous"


USAGE_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")

INSTRUCTIONS = """You annotate desired effects from pairs of Sokoban observations.
Use only the boards and the following exact vocabulary. Do not use tools, read
files, or consult other information. Return JSON matching the supplied schema.
The boards are arrays of equal-length strings. The top row and left column are
both zero, and coordinates are (row, column). Tile characters: # wall, - floor,
. empty goal, $ box, * box on goal, @ player, + player on goal. Boxes have no
individual identity. Supported predicates: box_at(row,col), player_at(row,col),
boxes_on_goals_at_least(count). Produce a nonempty conjunction of one to three
effects for each pair: EACH individual effect must be false on the start board
and true on the later board. Set preserve to an empty array. Optionally supply
a short natural-language instruction, or null. Describe effects only; do not
invent actions, claim a controller can achieve an effect, or add commentary.
Indices identify the pairs within this request only.
"""


def _predicate_schema() -> dict[str, Any]:
    coordinate = {"type": "integer", "minimum": 0}
    return {
        "anyOf": [{
            "type": "object",
            "properties": {
                "predicate": {
                    "type": "string",
                    "enum": [op]
                },
                "row": coordinate,
                "col": coordinate,
            },
            "required": ["predicate", "row", "col"],
            "additionalProperties": False,
        } for op in (SokobanPredicate.BOX_AT, SokobanPredicate.PLAYER_AT)] + [{
            "type": "object",
            "properties": {
                "predicate": {
                    "type": "string",
                    "enum": [SokobanPredicate.BOXES_ON_GOALS_AT_LEAST]
                },
                "count": {
                    "type": "integer",
                    "minimum": 0
                },
            },
            "required": ["predicate", "count"],
            "additionalProperties": False,
        }]
    }


def output_schema(size: int) -> dict[str, Any]:
    if not 1 <= size <= 4:
        raise ValueError("batch size must be between one and four")
    intent = {
        "type": "object",
        "properties": {
            "effects": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": _predicate_schema()
            },
            "preserve": {
                "type": "array",
                "maxItems": 0,
                "items": _predicate_schema()
            },
            "instruction": {
                "type": ["string", "null"]
            },
        },
        "required": ["effects", "preserve", "instruction"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "annotations": {
                "type": "array",
                "minItems": size,
                "maxItems": size,
                "items": {
                    "type": "object",
                    "properties": {
                        "pair_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": size - 1
                        },
                        "intent": intent,
                    },
                    "required": ["pair_index", "intent"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["annotations"],
        "additionalProperties": False,
    }


def build_prompt(records: list[dict[str, Any]]) -> str:
    if not 1 <= len(records) <= 4:
        raise ValueError("batch size must be between one and four")
    # Deliberately select only observations; never interpolate the source record.
    pairs = [{
        "pair_index": i,
        "start_observation": row["start_observation"],
        "later_observation": row["later_observation"]
    } for i, row in enumerate(records)]
    return INSTRUCTIONS + "\nPairs:\n" + json.dumps(pairs, separators=(",", ":")) + "\n"


def inspect_events(path: Path) -> tuple[list[str], int]:
    """Reject all item types other than text or reasoning, including tool use."""
    errors: list[str] = []
    count = 0
    for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"event_{line_no}_not_json")
            continue
        if not isinstance(event, dict):
            errors.append(f"event_{line_no}_not_object")
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "error":
            errors.append("codex_item:error")
        elif isinstance(item, dict) and item.get("type") not in {"agent_message", "reasoning"}:
            errors.append(f"disallowed_item_type:{item.get('type')}")
        if event.get("type") in {"error", "turn.failed"}:
            errors.append(f"codex_event:{event['type']}")
    if count == 0:
        errors.append("empty_event_stream")
    return errors, count


def extract_usage(path: Path) -> tuple[dict[str, int | None], UsageStatus]:
    """Read reported turn usage without inventing counts for missing fields."""
    empty: dict[str, int | None] = dict.fromkeys(USAGE_FIELDS)
    completed = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "turn.completed":
            completed.append(event)
    if not completed:
        return empty, UsageStatus.NOT_REPORTED
    if len(completed) != 1:
        return empty, UsageStatus.AMBIGUOUS
    usage = completed[0].get("usage")
    if not isinstance(usage, dict):
        return empty, UsageStatus.NOT_REPORTED
    counts: dict[str, int | None] = {}
    for field in USAGE_FIELDS:
        value = usage.get(field)
        counts[field] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    return counts, UsageStatus.COMPLETE if all(count is not None for count in counts.values()) else UsageStatus.PARTIAL


def total_usage(results: list[dict[str, Any]]) -> dict[str, int | None]:
    """Sum attempts only when each attempt reported that token count."""
    totals: dict[str, int | None] = {}
    for field in USAGE_FIELDS:
        counts = [result.get("usage", {}).get(field) for result in results]
        totals[field] = sum(counts) if counts and all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in counts) else None
    return totals


def validate_answer(records: list[dict[str, Any]], answer: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Check syntax and every effect; preservation is required to be empty."""
    errors: list[str] = []
    if not isinstance(answer, dict) or set(answer) != {"annotations"}:
        return [], ["answer_must_have_annotations_only"]
    rows = answer["annotations"]
    if not isinstance(rows, list) or len(rows) != len(records):
        return [], ["wrong_annotation_count"]
    by_index: dict[int, Any] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"pair_index", "intent"}:
            errors.append("malformed_annotation")
            continue
        index = row["pair_index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(records) or index in by_index:
            errors.append("invalid_or_duplicate_pair_index")
            continue
        by_index[index] = row["intent"]
    checked = []
    for index, record in enumerate(records):
        if index not in by_index:
            errors.append(f"pair_{index}:missing")
            continue
        intent = by_index[index]
        if not isinstance(intent, dict) or set(intent) != {"effects", "preserve", "instruction"}:
            errors.append(f"pair_{index}:invalid_intent_fields")
            continue
        if intent["preserve"] != []:
            errors.append(f"pair_{index}:preserve_must_be_empty")
        if intent["instruction"] is not None and not isinstance(intent["instruction"], str):
            errors.append(f"pair_{index}:invalid_instruction")
        effects = intent["effects"]
        if not isinstance(effects, list) or not 1 <= len(effects) <= 3:
            errors.append(f"pair_{index}:invalid_effect_count")
            continue
        # No preservation predicates are allowed in this protocol. Thus the
        # two endpoint states suffice for the checked effect conditions.
        try:
            reasons = validate_intent(intent, [record["start_observation"], record["later_observation"]])
            errors.extend(f"pair_{index}:{reason}" for reason in reasons)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            errors.append(f"pair_{index}:checker_error:{type(exc).__name__}:{exc}")
        checked.append({"record_id": record["record_id"], "intent": intent})
    return checked, sorted(set(errors))


def _sha(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _kill_group(proc: subprocess.Popen[Any]) -> None:
    """Stop the worker's group before reaping its leader, preventing PID reuse."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic() + 3
    while time.monotonic() < end:
        try:
            exited = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return
        if exited is not None:
            break
        time.sleep(0.05)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _proc_start(pid: int) -> int:
    """Read Linux process start ticks, guarding the watchdog against PID reuse."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    # The parenthesised comm field may contain spaces; the rest starts at field 3.
    return int(stat.rsplit(") ", 1)[1].split()[19])


def _register(registry: Path | None, event: dict[str, Any]) -> None:
    if registry is not None:
        with registry.open("a") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())


def freeze_selection(source: Path, destination: Path, count: int, seed: int) -> dict[str, Any]:
    """Preselect by layout in round-robin order, before any model response."""
    if count < 1 or destination.exists():
        raise ValueError("selection size must be positive and destination must not yet exist")
    source_bytes = source.read_bytes()
    groups: dict[str, list[dict[str, Any]]] = {}
    for line in source_bytes.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("split") != "train"
                or row.get("annotation_provenance", {}).get("kind") != "local_exact_difference"):
            continue
        selected = {
            key: row[key] for key in ("record_id", "layout_id", "split", "start_observation", "later_observation")
        }
        groups.setdefault(selected["layout_id"], []).append(selected)
    if not groups:
        raise ValueError("no training examples in source")

    def rank(value: str) -> str:
        return _sha(f"{seed}:{value}")

    boards = sorted(groups, key=rank)
    for group in groups.values():
        group.sort(key=lambda row: rank(row["record_id"]))
    selection: list[dict[str, Any]] = []
    while len(selection) < count and any(groups.values()):
        for board in boards:
            if groups[board] and len(selection) < count:
                selection.append(groups[board].pop(0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selection))
    os.replace(temporary, destination)
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selection_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "seed": seed,
        "requested_count": count,
        "actual_count": len(selection),
        "split": "train",
        "selection_rule": "only local_exact_difference, stable seed-hash order, round-robin over layouts",
        "record_ids": [row["record_id"] for row in selection],
        "first_record_is_pilot": True
    }
    _atomic_json(destination.with_suffix(destination.suffix + ".manifest.json"), manifest)
    return manifest


def merge_accepted(source: Path, summaries_dir: Path, destination: Path) -> dict[str, Any]:
    """Overlay independently checked model labels on train records only.

    The raw answer is retained.  Unchecked free-form instructions are excluded
    from the training target because the predicate checker cannot prove their
    semantic equivalence to the structured effects.
    """
    if destination.exists():
        raise ValueError("refusing to overwrite an existing merged dataset")
    source_bytes = source.read_bytes()
    records = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    lookup = {row["record_id"]: row for row in records}
    if len(lookup) != len(records):
        raise ValueError("source contains duplicate record IDs")
    replaced: set[str] = set()
    accepted_summaries = []
    for path in sorted(summaries_dir.glob("batch_*_summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("status") != AnnotationStatus.ACCEPTED:
            continue
        attempts = int(summary["provenance"]["attempts"])
        batch_name = path.name.removesuffix("_summary.json")
        attempt_dir = summaries_dir / f"{batch_name}_attempt_{attempts}"
        result = json.loads((attempt_dir / "result.json").read_text())
        if result["errors"] or result["accepted"] != summary["accepted"]:
            raise ValueError(f"inconsistent accepted summary: {path}")
        model = result.get("model")
        command = result.get("command")
        if not isinstance(model,
                          str) or model not in set(AnnotatorModel) or summary["provenance"].get("model") != model:
            raise ValueError(f"annotation model provenance does not match: {path}")
        if not isinstance(command, list) or command.count("-m") != 1:
            raise ValueError(f"annotation command lacks an unambiguous model: {path}")
        model_index = command.index("-m") + 1
        if model_index >= len(command) or command[model_index] != model:
            raise ValueError(f"annotation command model does not match: {path}")
        effort = result.get("reasoning_effort")
        if ((effort is not None and (not isinstance(effort, str) or effort not in set(ReasoningEffort)))
                or effort != summary["provenance"].get("reasoning_effort")):
            raise ValueError(f"annotation reasoning effort provenance does not match: {path}")
        effort_flags = [
            command[i + 1]
            for i, part in enumerate(command[:-1])
            if part == "-c" and isinstance(command[i + 1], str) and command[i + 1].startswith("model_reasoning_effort=")
        ]
        expected_effort_flags = [] if effort is None else [f'model_reasoning_effort="{effort}"']
        if effort_flags != expected_effort_flags:
            raise ValueError(f"annotation command reasoning effort does not match: {path}")
        for name in ("answer.json", "events.jsonl", "request.txt", "schema.json"):
            if not (attempt_dir / name).is_file():
                raise ValueError(f"missing raw provenance file {attempt_dir / name}")
        if (_sha((attempt_dir / "request.txt").read_text()) != result["prompt_sha256"] or _sha(
            (attempt_dir / "schema.json").read_text()) != result["schema_sha256"]):
            raise ValueError(f"prompt or schema hash does not match: {path}")
        event_errors, _ = inspect_events(attempt_dir / "events.jsonl")
        usage, usage_status = extract_usage(attempt_dir / "events.jsonl")
        if (("usage" in result or "usage_status" in result)
                and (result.get("usage") != usage or result.get("usage_status") != usage_status)):
            raise ValueError(f"annotation usage does not match raw event stream: {path}")
        batch_ids = summary["selected_record_ids"]
        if any(record_id not in lookup for record_id in batch_ids):
            raise ValueError(f"summary contains unknown record ID: {path}")
        checked, validation_errors = validate_answer([lookup[record_id] for record_id in batch_ids],
                                                     json.loads((attempt_dir / "answer.json").read_text()))
        if event_errors or validation_errors or checked != result["accepted"]:
            raise ValueError(f"raw trace or annotation failed revalidation: {path}")
        for accepted in result["accepted"]:
            record_id = accepted["record_id"]
            row = lookup.get(record_id)
            if row is None or record_id in replaced or row["split"] != "train":
                raise ValueError(f"unmatched, duplicate or non-training record: {record_id}")
            if row.get("annotation_provenance", {}).get("kind") != "local_exact_difference":
                raise ValueError("can only overlay locally labelled training records")
            raw_intent = accepted["intent"]
            if raw_intent.get("preserve") != []:
                raise ValueError("this annotation protocol only permits empty preserve")
            reasons = validate_intent(raw_intent, [row["start_observation"], row["later_observation"]])
            if reasons:
                raise ValueError(f"invalid accepted annotation {record_id}: {reasons}")
            intent = {**raw_intent, "instruction": None}
            operations = {effect["predicate"] for effect in intent["effects"]}
            if operations == {SokobanPredicate.PLAYER_AT}:
                effect_type = EffectType.WALK_ONLY
            elif SokobanPredicate.BOXES_ON_GOALS_AT_LEAST in operations:
                effect_type = EffectType.GOAL_CONJUNCTION if len(intent["effects"]) > 1 else EffectType.GOAL_PROGRESS
            else:
                effect_type = EffectType.BOX_CONJUNCTION if len(intent["effects"]) > 1 else EffectType.BOX
            row["intent"] = intent
            row["effect_type"] = effect_type
            row["annotation_provenance"] = {
                "kind": "fresh_isolated_codex_annotation",
                "model": model,
                "reasoning_effort": effort,
                "credential_source": result.get("credential_source", "unknown"),
                "usage": usage,
                "usage_status": usage_status,
                "usage_scope": "entire batch; do not sum across individual records",
                "auth_type": result.get("auth_type", "unknown"),
                "prompt_version": PROMPT_VERSION,
                "zero_shot": True,
                "prompt_sha256": result["prompt_sha256"],
                "schema_sha256": result["schema_sha256"],
                "batch_size": result["batch_size"],
                "batch_coupling_possible": result["batch_coupling_possible"],
                "attempts": attempts,
                "cache_hit": False,
                "raw_answer": str((attempt_dir / "answer.json").resolve()),
                "event_stream": str((attempt_dir / "events.jsonl").resolve()),
                "instruction_excluded_from_training": raw_intent.get("instruction") is not None,
            }
            row["validation"] = {
                "accepted": True,
                "each_effect_false_initially_true_finally": True,
                "preservation_checked_at_every_recorded_state": True,
                "rejection_reasons": []
            }
            replaced.add(record_id)
        accepted_summaries.append(str(path.resolve()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records))
    os.replace(temporary, destination)
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "merged_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "accepted_model_annotations": len(replaced),
        "record_ids": sorted(replaced),
        "accepted_summaries": accepted_summaries,
        "other_records": "retain their original, explicitly labelled provenance"
    }
    _atomic_json(destination.with_suffix(destination.suffix + ".manifest.json"), manifest)
    return manifest


def attempt(records: list[dict[str, Any]],
            directory: Path,
            timeout: float,
            deadline_monotonic: float,
            registry: Path | None = None,
            api_key_env: str | None = None,
            model: str = MODEL,
            reasoning_effort: str | None = None,
            api_key_dotenv: Path | None = None) -> dict[str, Any]:
    """Make one isolated call; save request, schema, trace, stderr, raw answer."""
    if model not in set(AnnotatorModel):
        raise ValueError(f"Unsupported annotation model: {model}")
    if reasoning_effort is not None and reasoning_effort not in set(ReasoningEffort):
        raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
    if api_key_env is not None and api_key_dotenv is not None:
        raise ValueError("Select at most one API credential source")
    directory.mkdir(parents=True, exist_ok=False)
    prompt = build_prompt(records)
    schema_text = json.dumps(output_schema(len(records)), sort_keys=True, indent=2) + "\n"
    (directory / "request.txt").write_text(prompt)
    (directory / "schema.json").write_text(schema_text)
    answer_path = directory / "answer.json"
    trace_path = directory / "events.jsonl"
    command = [
        "codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "-c",
        'shell_environment_policy.inherit="none"', "-m", model, "--output-schema",
        str((directory / "schema.json").resolve()), "-o",
        str(answer_path.resolve()), "--json", "-"
    ]
    if reasoning_effort is not None:
        command[2:2] = ["-c", f'model_reasoning_effort="{reasoning_effort}"']
    worker_env = os.environ.copy()
    credential_source = "inherited_codex_api_key" if worker_env.get("CODEX_API_KEY") else "saved_codex_auth"
    if api_key_dotenv is not None:
        from dotenv import dotenv_values

        key = dotenv_values(api_key_dotenv).get("OPENAI_API_KEY")
        if key is None or not key.strip():
            raise ValueError("Configured dotenv file contains no OPENAI_API_KEY")
        worker_env["CODEX_API_KEY"] = key.strip()
        credential_source = "dotenv"
    elif api_key_env is not None:
        key = worker_env.get(api_key_env)
        if not key:
            raise ValueError(f"API credential environment variable is missing: {api_key_env}")
        worker_env["CODEX_API_KEY"] = key
        credential_source = "environment"
        if api_key_env != "CODEX_API_KEY":
            worker_env.pop(api_key_env, None)
    worker_env.pop("OPENAI_API_KEY", None)
    auth_type = "api_key" if worker_env.get("CODEX_API_KEY") else "saved_codex_auth"
    result: dict[str, Any] = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_version": PROMPT_VERSION,
        "batch_size": len(records),
        "batch_coupling_possible": len(records) > 1,
        "prompt_sha256": _sha(prompt),
        "schema_sha256": _sha(schema_text),
        "command": command,
        "timeout_seconds": timeout,
        "auth_type": auth_type,
        "credential_source": credential_source,
        "start_monotonic": time.monotonic(),
        "return_code": None,
        "timed_out": False,
        "errors": [],
        "accepted": [],
        "usage": dict.fromkeys(USAGE_FIELDS),
        "usage_status": UsageStatus.NOT_REPORTED,
    }
    stop_monotonic = min(result["start_monotonic"] + timeout, deadline_monotonic)
    available = stop_monotonic - time.monotonic()
    if available <= 0:
        result["errors"] = ["central_deadline_elapsed"]
        _atomic_json(directory / "result.json", result)
        return result
    try:
        with tempfile.TemporaryDirectory(prefix="carl-intent-annotation-", dir="/tmp") as cwd, \
                trace_path.open("w") as trace, (directory / "stderr.txt").open("w") as stderr:
            # Gate exec until the process group has been registered.  If this
            # wrapper disappears before registration, EOF closes the gate and
            # the child exits instead of starting an untracked Codex process.
            read_fd, write_fd = os.pipe()
            gate = ("import os,sys; fd=int(sys.argv[1]); token=os.read(fd,1); os.close(fd); "
                    "sys.exit(125) if token!=b'1' else os.execvp(sys.argv[2],sys.argv[2:])")
            try:
                proc = subprocess.Popen([sys.executable, "-c", gate, str(read_fd), *command],
                                        cwd=cwd,
                                        stdin=subprocess.PIPE,
                                        stdout=trace,
                                        stderr=stderr,
                                        text=True,
                                        start_new_session=True,
                                        pass_fds=(read_fd,),
                                        env=worker_env)
            except BaseException:
                os.close(write_fd)
                raise
            finally:
                os.close(read_fd)
            result["pid"] = proc.pid
            result["process_group"] = proc.pid
            try:
                start_ticks = _proc_start(proc.pid)
                result["proc_start"] = start_ticks
                _register(
                    registry, {
                        "phase": WorkerPhase.STARTED,
                        "pid": proc.pid,
                        "pgid": proc.pid,
                        "proc_start": start_ticks,
                        "stop_monotonic": stop_monotonic,
                        "output": str(directory.resolve())
                    })
                remaining = stop_monotonic - time.monotonic()
                if remaining <= 0:
                    result["timed_out"] = True
                    _kill_group(proc)
                else:
                    os.write(write_fd, b"1")
                    os.close(write_fd)
                    write_fd = -1
                    try:
                        proc.communicate(prompt, timeout=remaining)
                    except subprocess.TimeoutExpired:
                        result["timed_out"] = True
                        _kill_group(proc)
                result["return_code"] = proc.returncode
            except BaseException:
                _kill_group(proc)
                raise
            finally:
                if write_fd >= 0:
                    os.close(write_fd)
                if "proc_start" in result:
                    _register(registry, {
                        "phase": WorkerPhase.FINISHED,
                        "pid": proc.pid,
                        "proc_start": result["proc_start"]
                    })
    except OSError as exc:
        result["errors"].append(f"launch_error:{type(exc).__name__}:{exc}")
    result["end_monotonic"] = time.monotonic()
    result["elapsed_seconds"] = result["end_monotonic"] - result["start_monotonic"]
    if result["timed_out"]:
        result["errors"].append("timeout")
    if result["return_code"] not in (None, 0):
        result["errors"].append(f"exit_code:{result['return_code']}")
    if trace_path.exists():
        trace_errors, count = inspect_events(trace_path)
        result["errors"].extend(trace_errors)
        result["event_count"] = count
        result["usage"], result["usage_status"] = extract_usage(trace_path)
    else:
        result["errors"].append("missing_event_stream")
    if answer_path.exists():
        try:
            answer = json.loads(answer_path.read_text())
            accepted, errors = validate_answer(records, answer)
            result["errors"].extend(errors)
            if not result["errors"]:
                result["accepted"] = accepted
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            result["errors"].append(f"invalid_json:{exc}")
    else:
        result["errors"].append("missing_answer")
    _atomic_json(directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, help="Frozen JSONL selection; must consist only of training records")
    parser.add_argument("--freeze-from",
                        type=Path,
                        help="Freeze --selection from this checked dataset and exit without a model call")
    parser.add_argument("--selection-count", type=int, default=40)
    parser.add_argument("--selection-seed", type=int, default=20261114)
    parser.add_argument("--merge-from",
                        type=Path,
                        help="Overlay accepted checked annotations from --summaries and exit")
    parser.add_argument("--merge-output", type=Path)
    parser.add_argument("--summaries", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--model", choices=list(AnnotatorModel), default=MODEL)
    parser.add_argument("--reasoning-effort",
                        choices=list(ReasoningEffort),
                        help="Override reasoning effort for this worker; omit to inherit the Codex configuration")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--retry", type=int, default=0, choices=[0, 1])
    parser.add_argument("--clock", type=Path)
    parser.add_argument("--registry", type=Path)
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--api-key-env",
                             help="Map this existing environment variable to CODEX_API_KEY for the worker only")
    credentials.add_argument("--api-key-dotenv",
                             type=Path,
                             help="Read OPENAI_API_KEY from this dotenv file for the worker only")
    args = parser.parse_args()
    if args.freeze_from is not None:
        if args.selection is None or args.merge_from is not None:
            parser.error("--freeze-from needs --selection and cannot be combined with --merge-from")
        print(json.dumps(freeze_selection(args.freeze_from, args.selection, args.selection_count, args.selection_seed)))
        return
    if args.merge_from is not None:
        if args.merge_output is None or args.summaries is None:
            parser.error("--merge-from requires --merge-output and --summaries")
        print(json.dumps(merge_accepted(args.merge_from, args.summaries, args.merge_output)))
        return
    if args.selection is None or args.output is None or args.clock is None:
        parser.error("--selection, --output and --clock are required for a model call")
    records = [json.loads(line) for line in args.selection.read_text().splitlines() if line.strip()]
    if any(record.get("split") != "train" for record in records):
        raise ValueError("Annotation selection must use train split only")
    if args.offset < 0 or args.offset + args.batch_size > len(records):
        raise ValueError("selected batch lies outside frozen selection")
    batch = records[args.offset:args.offset + args.batch_size]
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.api_key_env is not None and not os.environ.get(args.api_key_env):
        parser.error("the requested API credential environment variable is not set")
    clock = json.loads(args.clock.read_text())
    # Annotation is expensive work, so do not start it after 3 h 45 min.
    deadline = min(float(clock["deadline_monotonic"]), float(clock["start_monotonic"]) + 13500)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for number in range(args.retry + 1):
        result = attempt(batch, args.output / f"batch_{args.offset:04d}_attempt_{number + 1}", args.timeout, deadline,
                         args.registry, args.api_key_env, args.model, args.reasoning_effort, args.api_key_dotenv)
        results.append(result)
        if not result["errors"] or "central_deadline_elapsed" in result["errors"]:
            break
    final = results[-1]
    provenance = {
        "source": "isolated_codex",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_version": PROMPT_VERSION,
        "auth_type": final["auth_type"],
        "credential_source": final["credential_source"],
        "attempts": len(results),
        "batched_pairs": len(batch),
        "batch_coupling_possible": len(batch) > 1
    }
    _atomic_json(
        args.output / f"batch_{args.offset:04d}_summary.json", {
            "selected_record_ids": [record["record_id"] for record in batch],
            "provenance": provenance,
            "results": [result["errors"] for result in results],
            "usage": total_usage(results),
            "usage_by_attempt": [{
                "usage": result["usage"],
                "usage_status": result["usage_status"]
            } for result in results],
            "accepted": final["accepted"],
            "status": AnnotationStatus.ACCEPTED if not final["errors"] else AnnotationStatus.LOCAL_FALLBACK_REQUIRED,
        })
    if final["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
