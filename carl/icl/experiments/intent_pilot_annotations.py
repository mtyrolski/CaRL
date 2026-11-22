"""Blinded packet collection and independent intent dataset verification."""

import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carl.icl.api import write_json
from carl.icl.experiments.intent_pilot import verify
from carl.icl.experiments.intent_pilot_annotation_sources import craftax_episodes, sokoban_episodes
from carl.icl.intent_annotations import AnnotationRequest, SegmentSelection, UnlabelledAnnotation, prepare_annotations
from carl.icl.intent_domain import DOMAIN_SPEC, annotation_digest, annotation_json
from carl.icl.intents.generation.data import file_sha256, parse_json, read_records
from carl.icl.planning.craftax_tasks import CraftaxIntent, decode_intent, predicate_holds, render_instruction
from carl.icl.types import JSONObject


PROMPT = (
    "Given the public task, two observations and the supplied domain schema, propose a useful achieved change as "
    "typed effects and, when justified, a preservation condition worth checking. Use only supplied predicates and "
    "valid arguments. Do not infer continuous truth from endpoints. Do not provide actions, success probabilities, "
    "durations or scale. Return packet_id, effects, preserve and decision=propose; use decision=abstain with a brief "
    "reason when the observations do not support a useful proposal."
)
ANNOTATOR = {"name": "Codex", "kind": "model_annotation_in_this_session", "family": "GPT-6", "version": None}
PROMPT_SHA256 = annotation_digest(PROMPT.encode())

VOCABULARIES: dict[str, JSONObject] = {
    "sokoban": {
        "coordinates": DOMAIN_SPEC.coordinates, "tiles": dict(DOMAIN_SPEC.tile_meanings),
        "predicates": [{"predicate": item.name, "arguments": {key: "integer" for key in item.arguments},
                        "meaning": item.meaning} for item in DOMAIN_SPEC.predicates],
    },
    "craftax": {
        "predicates": [
            {"predicate": "inventory_at_least", "item": [
                "wood", "stone", "coal", "iron", "diamond", "sapling", "bow", "arrows", "torches",
                "ruby", "sapphire", "books",
            ], "count": "positive integer"},
            {"predicate": "tool_tier_at_least", "tool": ["pickaxe", "sword"], "tier": "integer from 1 to 4"},
        ],
    },
}


def _observation(domain: str, visible: Any) -> Any:
    # Inventory and facing are a projection of the native Craftax visible observation.
    return visible if domain == "sokoban" else {key: visible[key] for key in ("inventory", "facing")}


def _packet(domain: str, task: str, before: Any, after: Any) -> JSONObject:
    content = {
        "domain": domain, "task_goal": task, "start_observation": _observation(domain, before),
        "end_observation": _observation(domain, after), "vocabulary": VOCABULARIES[domain],
    }
    return {"packet_id": annotation_digest(annotation_json(content).encode()), **content}


def _emit(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing artefact differs at {path}")
    else:
        path.write_text(content, encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    _emit(path, "".join(annotation_json(row) + "\n" for row in rows))


def collect(root: Path, domain: str) -> JSONObject:
    """Collect traces and prepare blinded packets before consuming any labels."""
    output = root / "annotations" / domain
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    private = output / "private"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    split = verify(root)["split_sha256"]
    trace_file = private / "episodes.json"
    if not trace_file.exists():
        traces = list(sokoban_episodes(root) if domain == "sokoban" else craftax_episodes(root, private))
        write_json(trace_file, traces)
    else:
        traces = json.loads(trace_file.read_text(encoding="utf-8"))
    packets, index = [], []
    for trace in traces:
        start = trace["start"]
        for end in trace["ends"]:
            packet = _packet(domain, trace["task_goal"], trace["observations"][start], trace["observations"][end])
            index.append({"packet_id": packet["packet_id"], "episode_id": trace["episode_id"], "start": start, "end": end})
            packets.append(packet)
    if len({packet["packet_id"] for packet in packets}) != len(packets):
        raise ValueError("Blinded packet collection contains identical inputs")
    _jsonl(output / "packets.jsonl", packets)
    _jsonl(private / "packet-index.jsonl", index)
    _emit(output / "annotation-prompt.txt", PROMPT)
    membership = {
        role: sorted({trace["group_id"] for trace in traces if trace["pilot_split"] == role})
        for role in ("train-support", "validation")
    }
    if len(membership["train-support"]) != 8 or set(membership["train-support"]) & set(membership["validation"]):
        raise ValueError("Annotation collection needs eight support groups and disjoint validation groups")
    files = [trace_file, private / "packet-index.jsonl", output / "packets.jsonl", output / "annotation-prompt.txt"]
    if domain == "craftax":
        files.append(private / "craftax-collection-plan.json")
    manifest = {
        "domain": domain, "base_split_sha256": split, "group_membership": membership,
        "locked_files": {str(file.relative_to(output)): file_sha256(file) for file in files},
        "annotation_prompt_sha256": PROMPT_SHA256, "annotator": ANNOTATOR,
        "packets": len(packets), "trajectory_episodes": len(traces),
    }
    _emit(output / "split_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"domain": domain, "output": str(output), "packets": len(packets), "episodes": len(traces),
            "groups": {role: len(groups) for role, groups in membership.items()}}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _response(line: str, packet_id: str) -> JSONObject | UnlabelledAnnotation:
    try:
        response = parse_json(line)
    except (TypeError, ValueError) as error:
        return UnlabelledAnnotation("malformed", f"invalid raw JSON: {error}", json_valid=False)
    if not isinstance(response, dict) or response.get("packet_id") != packet_id:
        return UnlabelledAnnotation("malformed", "response packet_id differs from its blinded packet")
    if response.get("decision") == "abstain" and set(response) == {"packet_id", "decision", "reason"}:
        reason = response["reason"]
        if isinstance(reason, str) and reason.strip():
            return UnlabelledAnnotation("abstained", reason.strip())
    if response.get("decision") != "propose" or set(response) != {"packet_id", "decision", "effects", "preserve"}:
        return UnlabelledAnnotation("malformed", "expected a typed proposal or a reasoned abstention")
    return {"effects": response["effects"], "preserve": response["preserve"], "instruction": None}


def _provenance(packet_id: str, trace: dict) -> JSONObject:
    return {
        "packet_id": packet_id, "target_kind": "partial_predicate_intent", "pilot_split": trace["pilot_split"],
        "independent_group_id": trace["group_id"],
        "label_provenance": {**ANNOTATOR, "prompt_sha256": PROMPT_SHA256},
        "trajectory_provenance": {"kind": trace["trajectory_provenance"]["kind"]},
    }


def _sokoban_audit(
    output: Path, traces: dict[str, dict], index: list[dict], answers: dict[str, JSONObject | UnlabelledAnnotation],
) -> list[dict]:
    selections = [SegmentSelection(ref["episode_id"], ref["start"], ref["end"]) for ref in index]

    def annotate(request: AnnotationRequest) -> object:
        packet = _packet("sokoban", DOMAIN_SPEC.task_goal, request.start_observation.rows, request.later_observation.rows)
        return answers[packet["packet_id"]]

    with TemporaryDirectory(dir=output / "private", prefix="rebuild-") as directory:
        root = Path(directory)
        source = root / "trajectories.json"
        write_json(source, {"trajectories": [{**trace["source"], "observations": trace["observations"]}
                                             for trace in traces.values()]})
        prepare_annotations(
            source, root / "verified", seed=2026112203, max_offset=24, annotator=annotate,
            annotator_id="codex-session", selections=selections,
        )
        rows = _rows(root / "verified/annotation_audit.jsonl")

    for row, ref in zip(rows, index, strict=True):
        trace = traces[ref["episode_id"]]
        row.update(_provenance(ref["packet_id"], trace))
        evidence = row["evidence"]
        evidence["start_snapshot_ref"] = trace["snapshots"][ref["start"]]
        evidence["original_endpoint_ref"] = trace["snapshots"][ref["end"]]
        hit = evidence["first_verified_hit_offset"]
        evidence["first_verified_hit_ref"] = trace["snapshots"][ref["start"] + hit] if hit is not None else None
    return rows


def _craftax_hit(intent: CraftaxIntent, facts: list[dict], steps: list[dict]) -> int | UnlabelledAnnotation | tuple[str, str]:
    if any(len({annotation_json(predicate.model_dump()) for predicate in predicates}) != len(predicates)
           for predicates in (intent.effects, intent.preserve)):
        return UnlabelledAnnotation("malformed", "duplicate predicate within the proposal")
    endpoints = [(predicate_holds(effect, facts[0]), False, "effect already holds at start") for effect in intent.effects]
    endpoints.extend((predicate_holds(effect, facts[-1]), True, "effect false at original endpoint")
                     for effect in intent.effects)
    for observed, expected, reason in endpoints:
        if observed is not None and observed != expected:
            return "rejected", reason
    if any(observed is None for observed, _, _ in endpoints):
        return "unknown", "start or original endpoint effect evidence is unavailable"
    hit = next(i for i, inventory in enumerate(facts)
               if all(predicate_holds(effect, inventory) is True for effect in intent.effects))
    constraints = [(i, predicate_holds(predicate, inventory)) for i, inventory in enumerate(facts[:hit + 1])
                   for predicate in intent.preserve]
    false_at = next((i for i, observed in constraints if observed is False), None)
    if false_at is not None:
        return "rejected", f"preservation condition false at fragment state {false_at}"
    if any(observed is None for _, observed in constraints) or any(step["native_ticks"] != 1 for step in steps[:hit]):
        return "unknown", "required per-tick evidence is unavailable through the first hit"
    return hit


def _craftax_audit(
    traces: dict[str, dict], index: list[dict], answers: dict[str, JSONObject | UnlabelledAnnotation],
) -> list[dict]:
    records, seen = [], set()
    for ref in index:
        trace = traces[ref["episode_id"]]
        packet_id = ref["packet_id"]
        start, end = ref["start"], ref["end"]
        snapshots = trace["snapshots"]
        context = {"start_observation": trace["observations"][start], "task_goal": trace["task_goal"]}
        context_id = annotation_digest(annotation_json(context).encode())
        row = {
            "schema_version": "craftax-da-v1", "intent_version": "craftax-intent-v1", "context_id": context_id,
            "source_id": trace["episode_id"], "task_group": trace["group_id"], "world_id": trace["world_id"],
            "split": "train" if trace["pilot_split"] == "train-support" else "validation",
            **context, **_provenance(packet_id, trace),
            "evidence": {
                "start_index": start, "original_end_index": end, "start_snapshot_ref": snapshots[start],
                "original_endpoint_ref": snapshots[end], "original_end_observation": trace["observations"][end],
                "first_verified_hit_ref": None,
            },
        }
        answer = answers[packet_id]
        if isinstance(answer, UnlabelledAnnotation):
            decision: int | UnlabelledAnnotation | tuple[str, str] = answer
        else:
            try:
                intent = decode_intent(answer)
            except (TypeError, ValueError) as error:
                decision = UnlabelledAnnotation("malformed", f"invalid Craftax intent: {error}")
            else:
                row["intent"] = intent.model_dump(mode="json")
                decision = _craftax_hit(intent, trace["facts"][start:end + 1], trace["steps"][start:end])
        validation: JSONObject
        if isinstance(decision, UnlabelledAnnotation):
            validation = {"status": decision.status, "reasons": [decision.reason], "json_valid": decision.json_valid}
        elif isinstance(decision, tuple):
            validation = {"status": decision[0], "reasons": [decision[1]], "json_valid": True}
        else:
            index_at_hit = start + decision
            payload = row["intent"]
            semantic = {field: sorted(map(annotation_json, payload[field])) for field in ("effects", "preserve")}
            identifier = annotation_digest(annotation_json([context_id, semantic]).encode())
            row["record_id"] = identifier
            if identifier in seen:
                validation = {"status": "duplicate", "reasons": ["duplicate semantic intent for the same context"],
                              "duplicate_of": identifier, "json_valid": True}
            else:
                seen.add(identifier)
                validation = {"status": "accepted", "reasons": [], "json_valid": True}
            row["instruction"] = render_instruction(decode_intent(payload))
            row["demonstration_offset"] = snapshots[index_at_hit]["elapsed_primitive_actions"] - snapshots[start]["elapsed_primitive_actions"]
            row["evidence"].update(
                first_verified_hit_ref=snapshots[index_at_hit],
                actions=trace["actions"][start:index_at_hit], observations=trace["observations"][start:index_at_hit + 1],
                native_ticks=sum(step["native_ticks"] for step in trace["steps"][start:index_at_hit]),
            )
        row["validation"] = {"accepted": validation["status"] == "accepted", **validation}
        records.append(row)
    return records


def build(root: Path, domain: str, *, verify_only: bool = False) -> JSONObject:
    """Verify frozen packets against native evidence and rebuild the checked dataset."""
    output = root / "annotations" / domain
    manifest_file = output / "split_manifest.json"
    manifest = _read(manifest_file)
    if manifest["base_split_sha256"] != verify(root)["split_sha256"] or manifest["annotation_prompt_sha256"] != PROMPT_SHA256:
        raise ValueError("Annotation collection has a different split or prompt")
    for name, digest in manifest["locked_files"].items():
        if file_sha256(output / name) != digest:
            raise ValueError(f"Frozen annotation input differs at {name}")
    traces = {row["episode_id"]: row for row in _read(output / "private/episodes.json")}
    packets, index = _rows(output / "packets.jsonl"), _rows(output / "private/packet-index.jsonl")
    raw_path = output / "annotations.raw.jsonl"
    answers = {}
    for packet, ref, line in zip(packets, index, raw_path.read_text(encoding="utf-8").splitlines(), strict=True):
        trace = traces[ref["episode_id"]]
        expected = _packet(domain, trace["task_goal"], trace["observations"][ref["start"]], trace["observations"][ref["end"]])
        if expected != packet or expected["packet_id"] != ref["packet_id"]:
            raise ValueError("Blinded packet or native evidence no longer matches its hash")
        answers[packet["packet_id"]] = _response(line, packet["packet_id"])
    audit = _sokoban_audit(output, traces, index, answers) if domain == "sokoban" else _craftax_audit(traces, index, answers)
    by_split: dict[str, Counter[str]] = {}
    accepted = []
    for row in audit:
        status = row["validation"]["status"]
        by_split.setdefault(row["pilot_split"], Counter())[status] += 1
        if status == "accepted":
            accepted.append({name: value for name, value in row.items() if name != "evidence"})
    files = {"D_A.jsonl": accepted, "annotation_audit.jsonl": audit}
    for name, rows in files.items():
        if verify_only and not (output / name).is_file():
            raise ValueError(f"Cannot verify missing output {name}")
        _jsonl(output / name, rows)
    if domain == "sokoban":
        read_records(output / "D_A.jsonl")
    else:
        for row in accepted:
            decode_intent(row["intent"])
    summary = {
        "domain": domain, "split_manifest_sha256": file_sha256(manifest_file),
        "annotator": ANNOTATOR, "annotation_prompt_sha256": PROMPT_SHA256,
        "raw_response_sha256": file_sha256(raw_path), "raw_responses": len(answers),
        "accepted_by_split": {role: counts["accepted"] for role, counts in by_split.items()},
        "outcomes_by_split": {role: dict(counts) for role, counts in by_split.items()},
        "file_sha256": {name: file_sha256(output / name) for name in files},
    }
    summary_path = output / "manifest.json"
    if verify_only and not summary_path.is_file():
        raise ValueError("Cannot verify the missing annotation dataset manifest")
    _emit(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
