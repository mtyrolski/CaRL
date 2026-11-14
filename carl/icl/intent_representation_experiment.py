"""Compare equivalent Sokoban intent output representations with frozen ICL.

Only the query's current board and task are supplied to the generator.  The
checked support examples are selected from training layouts before evaluation.
This experiment fixes the optional instruction to null in every representation;
the predicate and preservation arrays are represented without loss.
"""

import argparse
import hashlib
import json
import os
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_data import SokobanPredicate
from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import evaluation_records
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import generate
from carl.icl.intent_generator_experiment import initialise
from carl.icl.intent_generator_experiment import memory
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_in_context import select_support


class Representation(StrEnum):
    JSON = "json"
    FORMAL = "formal"
    NATURAL = "natural"


class SupportSet(StrEnum):
    ZERO = "zero"
    LEGACY = "legacy"
    BALANCED0 = "balanced0"
    BALANCED1 = "balanced1"
    BALANCED2 = "balanced2"


class Mode(StrEnum):
    FREEZE_SUPPORTS = "freeze-supports"
    INSPECT = "inspect-prompts"
    EVALUATE = "evaluate"


COMMON = ("You propose one achievable, non-trivial Sokoban intent using only the current board. "
          "Rows and columns are zero-based. # is wall, - is floor, . is goal, $ is box, "
          "* is box on goal, @ is player and + is player on goal. "
          "An intent has required effects and preservation conditions. All effects must hold "
          "together and each must be false on the current board. Every preservation condition "
          "must be true now and remain true. Boxes are indistinguishable. "
          "The only predicates are box at (row,column), player at (row,column), "
          "and at least a given number of boxes on goals. Return one intent, with no other text. ")

REPRESENTATION_INSTRUCTIONS = {
    Representation.JSON:
        'Return exactly a JSON object with keys "effects", "preserve", "instruction". '
        'Both effects and preserve are arrays; effects is nonempty. instruction must be null. '
        'Predicates are {"predicate":"box_at","row":R,"col":C}, '
        '{"predicate":"player_at","row":R,"col":C}, or '
        '{"predicate":"boxes_on_goals_at_least","count":N}.',
    Representation.FORMAL:
        "Return exactly E[effects] P[preserve]. Separate predicates in each list with semicolons; "
        "the preserve list may be empty. Write predicates only as box(R,C), player(R,C), or goals(N). "
        "For example E[box(2,3);goals(2)] P[] requires both effects and preserves nothing.",
    Representation.NATURAL:
        "Return exactly 'Achieve: effects. Preserve: conditions.' Join predicates with ' and '. "
        "If there are no preservation conditions write 'nothing'. "
        "Write predicates only as 'box at row R column C', 'player at row R column C', or "
        "'at least N boxes on goals'. Example: Achieve: box at row 2 column 3. Preserve: nothing.",
}


def _canonical(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"effects", "preserve", "instruction"}:
        raise ValueError("expected effects, preserve and instruction keys")
    if value["instruction"] is not None:
        raise ValueError("this experiment fixes instruction to null")
    result: dict[str, Any] = {"effects": [], "preserve": [], "instruction": None}
    for group in ("effects", "preserve"):
        items = value[group]
        if not isinstance(items, list) or (group == "effects" and not items):
            raise ValueError(f"{group} must be an array" + (" with at least one item" if group == "effects" else ""))
        for predicate in items:
            if not isinstance(predicate, dict):
                raise ValueError("predicate must be an object")
            name = predicate.get("predicate")
            if name == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST:
                fields = ("count",)
            elif name in (SokobanPredicate.BOX_AT, SokobanPredicate.PLAYER_AT):
                fields = ("row", "col")
            else:
                raise ValueError("unknown predicate")
            if set(predicate) != {"predicate", *fields}:
                raise ValueError("unexpected predicate fields")
            if any(not isinstance(predicate[key], int) or isinstance(predicate[key], bool) or predicate[key] < 0
                   for key in fields):
                raise ValueError("predicate arguments must be nonnegative integers")
            result[group].append({"predicate": str(name), **{key: predicate[key] for key in fields}})
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _render_predicate(predicate: dict[str, Any], representation: Representation) -> str:
    name = predicate["predicate"]
    if name == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST:
        return (f"goals({predicate['count']})" if representation == Representation.FORMAL else
                f"at least {predicate['count']} boxes on goals")
    subject = "box" if name == SokobanPredicate.BOX_AT else "player"
    if representation == Representation.FORMAL:
        return f"{subject}({predicate['row']},{predicate['col']})"
    return f"{subject} at row {predicate['row']} column {predicate['col']}"


def render_representation(representation: Representation | str, value: Any) -> str:
    """Serialize all predicates and arrays; reject unsupported instructions."""
    representation = Representation(representation)
    intent = _canonical(value)
    if representation == Representation.JSON:
        return json.dumps(intent, ensure_ascii=False, separators=(",", ":"))
    effects = [_render_predicate(item, representation) for item in intent["effects"]]
    preserve = [_render_predicate(item, representation) for item in intent["preserve"]]
    if representation == Representation.FORMAL:
        return f"E[{';'.join(effects)}] P[{';'.join(preserve)}]"
    return f"Achieve: {' and '.join(effects)}. Preserve: {' and '.join(preserve) if preserve else 'nothing'}."


def _parse_predicate(text: str, representation: Representation) -> dict[str, Any]:
    if representation == Representation.FORMAL:
        position = re.fullmatch(r"(box|player)\((\d+),(\d+)\)", text)
        goals = re.fullmatch(r"goals\((\d+)\)", text)
    else:
        position = re.fullmatch(r"(box|player) at row (\d+) column (\d+)", text)
        goals = re.fullmatch(r"at least (\d+) boxes on goals", text)
    if position:
        return {"predicate": "box_at" if position[1] == "box" else "player_at", "row": int(position[2]),
                "col": int(position[3])}
    if goals:
        return {"predicate": "boxes_on_goals_at_least", "count": int(goals[1])}
    raise ValueError("predicate does not match the specified representation")


def parse_representation(representation: Representation | str, raw: str) -> dict[str, Any]:
    """Strictly parse one output into the common JSON intent schema."""
    representation = Representation(representation)
    if not isinstance(raw, str):
        raise ValueError("output must be text")
    text = raw.strip()
    if representation == Representation.JSON:
        def reject_constant(value: str) -> None:
            raise ValueError(f"non-JSON constant {value}")
        try:
            return _canonical(json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if representation == Representation.FORMAL:
        match = re.fullmatch(r"E\[(.+?)\] P\[(.*?)\]", text)
        separator, empty = ";", ""
    else:
        match = re.fullmatch(r"Achieve: (.+?)\. Preserve: (.+?)\.", text)
        separator, empty = " and ", "nothing"
    if not match:
        raise ValueError("output does not match the specified representation")
    effects = [_parse_predicate(item, representation) for item in match[1].split(separator)]
    preserve = ([] if match[2] == empty else
                [_parse_predicate(item, representation) for item in match[2].split(separator)])
    return _canonical({"effects": effects, "preserve": preserve, "instruction": None})


def _whole_task(row: dict[str, Any]) -> bool:
    board = row["start_observation"]
    goals = sum(line.count(".") + line.count("*") + line.count("+") for line in board)
    return any(item["predicate"] == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST and item["count"] >= goals
               for item in row["intent"]["effects"])


def support_bank(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Prespecified train-only support selection; independent of test inputs."""
    candidates = sorted(
        (row for row in records if row.get("split") == DataSplit.TRAIN
         and row.get("validation", {}).get("accepted") is True
         and row.get("intent", {}).get("instruction") is None),
        key=lambda row: (row["layout_id"], row["record_id"]),
    )
    selected = {SupportSet.ZERO.value: [], SupportSet.LEGACY.value: select_support(records)}
    used_layouts: set[str] = set()
    for name in (SupportSet.BALANCED0, SupportSet.BALANCED1, SupportSet.BALANCED2):
        pair = []
        for effect_type in ("box", "goal_conjunction"):
            match = next((row for row in candidates if row.get("effect_type") == effect_type
                          and row["layout_id"] not in used_layouts and not _whole_task(row)), None)
            if match is None:
                raise ValueError(f"not enough distinct training layouts for {name} {effect_type}")
            _canonical(match["intent"])
            pair.append(match)
            used_layouts.add(match["layout_id"])
        selected[name.value] = pair
    return selected


def support_manifest(data: Path) -> dict[str, Any]:
    bank = support_bank(read_records(data))
    return {
        "schema_version": 1,
        "support_data_sha256": file_sha256(data),
        "selection": "legacy = existing deterministic support; balanced sets = sorted checked TRAIN "
                     "box and goal_conjunction examples on six distinct layouts, excluding full-goal labels; "
                     "zero = no examples. No test records or model outputs were used.",
        "caveat": "Balanced support variants were conceived after historical ICL results and are exploratory.",
        "support_sets": {
            name: [{"record_id": row["record_id"], "layout_id": row["layout_id"],
                    "effect_type": row.get("effect_type"), "whole_task": _whole_task(row)} for row in group]
            for name, group in bank.items()
        },
    }


def load_supports(data: Path, manifest_path: Path, name: SupportSet) -> list[dict[str, Any]]:
    expected = support_manifest(data)
    if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
        raise ValueError("support manifest differs from the prespecified train-only selection")
    return support_bank(read_records(data))[name.value]


def messages_for(record: dict[str, Any], support: list[dict[str, Any]],
                 representation: Representation | str) -> list[dict[str, str]]:
    representation = Representation(representation)
    query = prompt_messages(record)
    messages = [{"role": "system", "content": COMMON + REPRESENTATION_INSTRUCTIONS[representation]}]
    for example in support:
        if (example.get("split") != DataSplit.TRAIN or example.get("validation", {}).get("accepted") is not True
                or example["layout_id"] == record["layout_id"]):
            raise ValueError("support must be checked TRAIN data from a distinct layout")
        messages.extend([prompt_messages(example)[1],
                         {"role": "assistant", "content": render_representation(representation, example["intent"])}])
    messages.append(query[1])
    return messages


def _write_once(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise ValueError(f"refusing to overwrite a different frozen file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def prepare(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    from transformers import AutoTokenizer

    all_rows = read_records(args.data, args.split)
    ids = json.loads(Path(args.test_ids).read_text(encoding="utf-8"))
    if ids.get("dataset_sha256") not in (None, file_sha256(args.data)):
        raise ValueError("frozen IDs belong to a different dataset")
    records = evaluation_records(all_rows, args.test_ids, 0)
    if not records or len({r["record_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("empty selection or duplicate records")
    support = load_supports(Path(args.support_data), Path(args.support_manifest), SupportSet(args.support_set))
    if {row["layout_id"] for row in support} & {row["layout_id"] for row in records}:
        raise ValueError("support and evaluated layouts overlap")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.chat_template is None:
        raise ValueError("model tokenizer has no chat template")
    prompts = []
    for record in records:
        messages = messages_for(record, support, args.representation)
        options = {"add_generation_prompt": True, "enable_thinking": False}
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, **options)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        if len(tokens) + args.max_new_tokens > args.max_length:
            raise ValueError(f"complete prompt needs {len(tokens)} + {args.max_new_tokens} tokens; "
                             f"limit is {args.max_length}; no truncation")
        prompts.append({"record_id": record["record_id"], "layout_id": record["layout_id"],
                        "messages": messages, "rendered_prompt": rendered, "prompt_tokens": len(tokens)})
    prompt_path = Path(args.output).with_suffix(".prompts.jsonl")
    _write_once(prompt_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts))
    return records, prompts, prompt_path


def configuration(args: argparse.Namespace, prompt_path: Path) -> dict[str, Any]:
    adapter = Path(args.adapter) if args.adapter else None
    model = Path(args.model)
    if not model.is_dir():
        raise ValueError("use an existing, explicitly resolved local model snapshot")
    revision_metadata = model / "experiment_revision.json"
    if revision_metadata.exists():
        if json.loads(revision_metadata.read_text(encoding="utf-8")).get("revision") != args.revision:
            raise ValueError("local model revision metadata does not match the requested revision")
        revision_verification = "experiment_revision.json"
    elif re.fullmatch(r"[0-9a-f]{40}", model.name) and model.name == args.revision:
        revision_verification = "huggingface snapshot directory"
    else:
        raise ValueError("cannot verify local model revision from metadata or snapshot directory")
    return {
        "schema_version": 1,
        "variant": args.variant or f"{args.representation}_{args.support_set}",
        "representation": args.representation,
        "support_set": args.support_set,
        "model": str(model.resolve()),
        "revision": args.revision,
        "revision_verification": revision_verification,
        "model_weights_sha256": file_sha256(model / "model.safetensors"),
        "model_config_sha256": file_sha256(model / "config.json"),
        "tokenizer_sha256": file_sha256(model / "tokenizer.json"),
        "adapter": str(adapter.resolve()) if adapter else None,
        "adapter_weights_sha256": file_sha256(adapter / "adapter_model.safetensors") if adapter else None,
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json") if adapter else None,
        "data_sha256": file_sha256(args.data),
        "test_ids_sha256": file_sha256(args.test_ids),
        "support_data_sha256": file_sha256(args.support_data),
        "support_manifest_sha256": file_sha256(args.support_manifest),
        "prompts_sha256": file_sha256(prompt_path),
        "protocol_sha256": file_sha256(__file__),
        "generator_sha256": file_sha256(Path(__file__).with_name("intent_generator_experiment.py")),
        "support_selection_sha256": file_sha256(Path(__file__).with_name("intent_in_context.py")),
        "split": args.split,
        "dtype": args.dtype,
        "seed": args.seed,
        "threads": args.threads,
        "decoding": "greedy",
        "enable_thinking": False,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "require_cuda": args.require_cuda,
    }


def _parse_output(representation: str, output: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    try:
        return parse_representation(representation, output), True, None
    except ValueError as exc:
        return None, False, str(exc)


def prepare_output(output: Path, config: dict[str, Any], records: list[dict[str, Any]],
                   resume: bool) -> list[dict[str, Any]]:
    """Validate a partial run against exact settings, order, and raw outputs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = output.with_suffix(".run.json")
    if not resume and (output.exists() or metadata.exists()):
        raise FileExistsError("output already exists; use --resume for this exact run")
    if output.exists() and not metadata.exists():
        raise ValueError("cannot resume without a frozen run configuration")
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != config:
            raise ValueError("run configuration changed; refusing to resume")
    else:
        atomic_json(metadata, config)
    if not output.exists():
        return []
    raw = output.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        tail = raw[boundary:]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            saved = output.with_name(f"{output.name}.partial-{hashlib.sha256(tail).hexdigest()[:12]}")
            saved.write_bytes(tail)
            raw = raw[:boundary]
        else:
            raw += b"\n"
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, output)
    previous = read_records(output)
    if len(previous) > len(records):
        raise ValueError("output exceeds frozen selection")
    for row, record in zip(previous, records, strict=False):
        if (row.get("record_id") != record["record_id"] or row.get("layout_id") != record["layout_id"]
                or row.get("split") != config["split"] or row.get("variant") != config["variant"]
                or row.get("representation") != config["representation"]
                or row.get("support_set") != config["support_set"] or not isinstance(row.get("raw_output"), str)):
            raise ValueError("saved output does not match frozen selection or configuration")
        parsed, syntax_valid, _ = _parse_output(config["representation"], row["raw_output"])
        if row.get("syntax_valid") is not syntax_valid or row.get("parsed_intent") != parsed:
            raise ValueError("saved parse does not match raw output")
    return previous


def _write_partial(output: Path, count: int, planned: int, reason: str, error: BaseException | None = None) -> None:
    atomic_json(output.with_suffix(".partial-summary.json"), {
        "status": "stopped", "processed": count, "planned": planned, "reason": reason,
        "error_type": type(error).__name__ if error else None, "error": str(error) if error else None,
    })


def evaluate(args: argparse.Namespace) -> None:
    records, prompts, prompt_path = prepare(args)
    if args.mode == Mode.INSPECT:
        counts = [item["prompt_tokens"] for item in prompts]
        print(json.dumps({"status": "prepared", "n": len(records), "min_prompt_tokens": min(counts),
                          "max_prompt_tokens": max(counts), "max_new_tokens": args.max_new_tokens,
                          "max_length": args.max_length, "prompts_path": str(prompt_path)}, indent=2))
        return
    config = configuration(args, prompt_path)
    output = Path(args.output)
    previous = prepare_output(output, config, records, args.resume)
    count = len(previous)
    started = time.monotonic()
    try:
        if ((args.stop_monotonic is not None and time.monotonic() >= args.stop_monotonic)
                or (args.stop_epoch is not None and time.time() >= args.stop_epoch)):
            _write_partial(output, count, len(records), "deadline")
            raise SystemExit("deadline reached before model initialisation")
        if args.require_cuda:
            import torch as check_torch

            if not check_torch.cuda.is_available():
                raise RuntimeError("CUDA is required but unavailable; no silent CPU fallback")
        torch, tokenizer, model, device = initialise(args, adapter=args.adapter, train=False)
        with output.open("a", encoding="utf-8", buffering=1) as destination:
            for record, prompt in zip(records[count:], prompts[count:], strict=True):
                if ((args.stop_monotonic is not None and time.monotonic() >= args.stop_monotonic)
                        or (args.stop_epoch is not None and time.time() >= args.stop_epoch)):
                    _write_partial(output, count, len(records), "deadline")
                    raise SystemExit("deadline reached; partial predictions retained")
                raw, tokens, elapsed = generate(torch, model, tokenizer, record, args, device,
                                                messages=prompt["messages"])
                parsed, valid, error = _parse_output(args.representation, raw)
                row = {"record_id": record["record_id"], "layout_id": record["layout_id"], "split": args.split,
                       "variant": config["variant"], "representation": args.representation,
                       "support_set": args.support_set, "raw_output": raw, "syntax_valid": valid,
                       "parsed_intent": parsed, "syntax_error": error, "generated_tokens": tokens,
                       "generation_seconds": elapsed}
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
                os.fsync(destination.fileno())
                count += 1
                print(json.dumps({"record_id": record["record_id"], "syntax_valid": valid,
                                  "generation_seconds": elapsed}), flush=True)
    except Exception as exc:
        _write_partial(output, count, len(records), "infrastructure_error", exc)
        raise
    atomic_json(output.with_suffix(".summary.json"), {
        "status": "complete", "n": count, "resumed_records": len(previous),
        "elapsed_seconds": time.monotonic() - started, **memory(torch),
    })
    partial = output.with_suffix(".partial-summary.json")
    if partial.exists():
        old = json.loads(partial.read_text(encoding="utf-8"))
        old.update(final_status="complete", superseded_by=output.with_suffix(".summary.json").name)
        atomic_json(partial, old)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=list(Mode))
    parser.add_argument("--data")
    parser.add_argument("--support-data", required=True)
    parser.add_argument("--support-manifest")
    parser.add_argument("--test-ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--representation", choices=list(Representation))
    parser.add_argument("--support-set", choices=list(SupportSet))
    parser.add_argument("--variant")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--adapter")
    parser.add_argument("--split", choices=list(DataSplit), default=DataSplit.TEST)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--stop-epoch", type=float)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == Mode.FREEZE_SUPPORTS:
        manifest = support_manifest(Path(args.support_data))
        _write_once(Path(args.output), json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(args.output)
        return
    if not all((args.data, args.test_ids, args.support_manifest, args.representation, args.support_set,
                args.model, args.revision)):
        parser.error("data, test IDs, support manifest/set, representation, model and revision are required")
    if args.max_length < 1 or args.max_new_tokens < 1 or args.threads < 1:
        parser.error("length and thread limits must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
