"""Board-level generator evaluation and separately labelled bounded witnesses.

The bounded search is a deterministic solver, not an LLM or learned policy.
It is given only a proposed intent and its current board.  Any witness is
replayed with the repository's real Sokoban environment before being counted.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from carl.icl.intent_data import SokobanPredicate
from carl.icl.intent_data import predicate_value
from carl.icl.intent_data import validate_prediction

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Method(StrEnum):
    BASE = "base"
    ADAPTED = "adapted"
    IN_CONTEXT = "in_context"
    LOCAL_RULE = "local_rule"


class SearchStatus(StrEnum):
    WITNESS_FOUND = "witness_found"
    DEPTH_LIMIT = "depth_limit"
    NODE_LIMIT = "node_limit"
    TIME_LIMIT = "time_limit"
    EXHAUSTED = "exhausted"


class AssessmentStatus(StrEnum):
    EVALUATED = "evaluated"
    MISSING = "missing_infrastructure_output"


Coordinate = tuple[int, int]
Positions = frozenset[Coordinate]
State = tuple[Coordinate, Positions]
Record = dict[str, Any]


def board_parts(rows: list[str],) -> tuple[int, int, set[Coordinate], set[Coordinate], Positions, Coordinate]:
    if not rows or not rows[0]:
        raise ValueError("Invalid Sokoban board")
    height, width = len(rows), len(rows[0])
    walls = {(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "#"}
    goals = {(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in ".*+"}
    boxes = frozenset((r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in "$*")
    players = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in "@+"]
    if len(players) != 1 or any(len(row) != width for row in rows):
        raise ValueError("Invalid Sokoban board")
    return height, width, walls, goals, boxes, players[0]


def step(
    player: Coordinate,
    boxes: Positions,
    action: int,
    height: int,
    width: int,
    walls: set[Coordinate],
) -> State | None:
    dr, dc = DIRECTIONS[action]
    dest = player[0] + dr, player[1] + dc
    if not (0 <= dest[0] < height and 0 <= dest[1] < width) or dest in walls:
        return None
    if dest in boxes:
        beyond = dest[0] + dr, dest[1] + dc
        if (not (0 <= beyond[0] < height and 0 <= beyond[1] < width) or beyond in walls or beyond in boxes):
            return None
        boxes = frozenset((boxes - {dest}) | {beyond})
    return dest, boxes


def goal_satisfied(effects: list[Record], player: Coordinate, boxes: Positions, goals: set[Coordinate]) -> bool:
    for predicate in effects:
        if predicate["predicate"] == SokobanPredicate.BOX_AT:
            value = (predicate["row"], predicate["col"]) in boxes
        elif predicate["predicate"] == SokobanPredicate.PLAYER_AT:
            value = (predicate["row"], predicate["col"]) == player
        else:
            value = len(boxes & goals) >= predicate["count"]
        if not value:
            return False
    return True


def real_replay(rows: list[str], intent: Record, actions: list[int]) -> list[str]:
    """Verify candidate witness with the same real environment as the data."""
    import numpy as np

    from carl.icl.boards import from_rows
    from carl.icl.boards import to_rows
    from carl.icl.data_sources.sokoban import environment

    state = from_rows(rows)
    env = environment(state)
    if not all(predicate_value(p, rows) for p in intent.get("preserve", [])):
        raise AssertionError("Preservation false initially")
    for action in actions:
        if env.is_solved(state):
            raise AssertionError("Witness continues after task termination")
        new = np.asarray(env.next_state(state, action))
        if np.array_equal(new, state):
            raise AssertionError("Witness contains a failed or no-op action")
        state = new
        current = to_rows(state)
        if not all(predicate_value(p, current) for p in intent.get("preserve", [])):
            raise AssertionError("Witness violates preservation")
    reached = to_rows(state)
    if not all(predicate_value(p, reached) for p in intent["effects"]):
        raise AssertionError("Witness failed real environment check")
    return cast(list[str], reached)


def bounded_witness(
    rows: list[str],
    intent: Record,
    max_depth: int = 24,
    max_nodes: int = 10000,
    seconds: float = 1.0,
) -> Record:
    errors = validate_prediction(intent, rows)
    if errors:
        raise ValueError("; ".join(errors))
    height, width, walls, goals, boxes, player = board_parts(rows)
    initial = player, boxes
    frontier = deque([(initial, 0)])
    parents: dict[State, tuple[State, int] | None] = {initial: None}
    start = time.monotonic()
    expanded = 0
    primitive_edges_examined = 0
    depth_limited = False
    reason = SearchStatus.EXHAUSTED
    end_state: State | None = None
    while frontier:
        if time.monotonic() - start >= seconds:
            reason = SearchStatus.TIME_LIMIT
            break
        if expanded >= max_nodes:
            reason = SearchStatus.NODE_LIMIT
            break
        (player, boxes), depth = frontier.popleft()
        if depth >= max_depth:
            depth_limited = True
            continue
        # The actual environment ends the task when all boxes occupy goals.
        if boxes <= goals:
            continue
        expanded += 1
        for action in range(4):
            primitive_edges_examined += 1
            nxt = step(player, boxes, action, height, width, walls)
            if nxt is None or nxt in parents:
                continue
            next_player, next_boxes = nxt
            if not goal_satisfied(intent.get("preserve", []), next_player, next_boxes, goals):
                continue
            parents[nxt] = ((player, boxes), action)
            if goal_satisfied(intent["effects"], next_player, next_boxes, goals):
                end_state = nxt
                reason = SearchStatus.WITNESS_FOUND
                break
            frontier.append((nxt, depth + 1))
        if end_state is not None:
            break
    if reason == SearchStatus.EXHAUSTED and depth_limited:
        reason = SearchStatus.DEPTH_LIMIT
    search_seconds = time.monotonic() - start
    actions = None
    reached = None
    verification_seconds = None
    if end_state is not None:
        actions = []
        node = end_state
        while (parent := parents[node]) is not None:
            node, action = parent
            actions.append(action)
        actions.reverse()
        verification_start = time.monotonic()
        reached = real_replay(rows, intent, actions)
        verification_seconds = time.monotonic() - verification_start
    return {
        "status": reason,
        "witness_found": end_state is not None,
        "depth_limit": max_depth,
        "node_limit": max_nodes,
        "time_limit_seconds": seconds,
        "expanded_nodes": expanded,
        "primitive_edges_examined": primitive_edges_examined,
        "elapsed_seconds": search_seconds,
        "verification_seconds": verification_seconds,
        "witness_depth": len(actions) if actions is not None else None,
        "witness_actions": actions,
        "reached_observation": reached,
    }


def local_rule_with_stats(rows: list[str]) -> tuple[Record | None, Record]:
    """Pick an accessible push and measure the rule's local graph work.

    The input is only the current board. Walking reachability is computed on
    the unchanged current arrangement; no demonstration state is consulted.
    Edges count every tested direction, including blocked and repeated ones.
    """
    started = time.perf_counter()
    height, width, walls, goals, boxes, player = board_parts(rows)
    paths = {player: 0}
    queue = deque([player])
    expanded = edges = push_checks = goal_distance_checks = 0
    while queue:
        here = queue.popleft()
        expanded += 1
        for dr, dc in DIRECTIONS:
            edges += 1
            nxt = here[0] + dr, here[1] + dc
            if (0 <= nxt[0] < height and 0 <= nxt[1] < width and nxt not in walls and nxt not in boxes
                    and nxt not in paths):
                paths[nxt] = paths[here] + 1
                queue.append(nxt)
    choices = []
    for r, c in sorted(boxes):
        for dr, dc in DIRECTIONS:
            push_checks += 1
            stand, dest = (r - dr, c - dc), (r + dr, c + dc)
            if (stand in paths and 0 <= dest[0] < height and 0 <= dest[1] < width and dest not in walls
                    and dest not in boxes):
                goal_distance_checks += len(goals)
                nearest = min(
                    (abs(dest[0] - gr) + abs(dest[1] - gc) for gr, gc in goals),
                    default=0,
                )
                choices.append((dest not in goals, nearest, paths[stand], dest))
    if choices:
        _, _, _, (r, c) = min(choices)
        effect = {"predicate": SokobanPredicate.BOX_AT, "row": r, "col": c}
    elif len(paths) > 1:
        r, c = min((p for p in paths if p != player), key=lambda p: (paths[p], p))
        effect = {"predicate": SokobanPredicate.PLAYER_AT, "row": r, "col": c}
    else:
        effect = None
    intent = {"effects": [effect], "preserve": [], "instruction": None} if effect is not None else None
    stats = {"walk_graphs": 1, "walk_nodes_expanded": expanded, "walk_edges_examined": edges,
             "reachable_tiles": len(paths), "push_checks": push_checks, "legal_pushes": len(choices),
             "goal_distance_checks": goal_distance_checks, "elapsed_seconds": time.perf_counter() - started}
    return intent, stats


def local_rule(rows: list[str]) -> Record | None:
    """Return the deterministic current-board proposal, preserving the original API."""
    return local_rule_with_stats(rows)[0]


def test_contexts(records: list[Record], ids_path: Path | None = None) -> list[Record]:
    test = [row for row in records if row["split"] == "test"]
    if ids_path is None:
        return test
    ids = json.loads(ids_path.read_text(encoding="utf-8"))["record_ids"]
    by_id = {row["record_id"]: row for row in test}
    if (len(by_id) != len(test) or not isinstance(ids, list) or len(ids) != len(set(ids))
            or any(identifier not in by_id for identifier in ids)):
        raise ValueError("Frozen context IDs must be distinct test records")
    return [by_id[identifier] for identifier in ids]


def read_jsonl(path: str | Path) -> list[Record]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _reject_non_json(value: str) -> None:
    raise ValueError(f"non-JSON constant: {value}")


def check_raw_prediction(row: Record) -> None:
    """Make JSON validity a property of the saved raw text, never repair it."""
    if not isinstance(row.get("raw_output"), str) or not isinstance(row.get("json_valid"), bool):
        raise ValueError("Prediction is missing its raw output or JSON validity flag")
    try:
        parsed = json.loads(row["raw_output"], parse_constant=_reject_non_json)
    except ValueError:
        if row["json_valid"] or row.get("parsed") is not None:
            raise ValueError("Invalid raw JSON disagrees with saved prediction") from None
    else:
        if not row["json_valid"] or row.get("parsed") != parsed:
            raise ValueError("Parsed JSON disagrees with saved raw output")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions_for(
    path: Path | None,
    method: Method,
    contexts: list[Record],
    data_path: Path,
    ids_path: Path | None,
) -> tuple[dict[str, Record], Record | None]:
    """Reject mismatched, duplicate or extra predictions before pairing."""
    if not path or not path.exists():
        return {}, None
    expected = {row["record_id"]: row["layout_id"] for row in contexts}
    predictions: dict[str, Record] = {}
    for row in read_jsonl(path):
        record_id = cast(str, row.get("record_id"))
        if (record_id in predictions or record_id not in expected or row.get("layout_id") != expected[record_id]
                or row.get("split") != "test" or row.get("method") != method):
            raise ValueError(f"Prediction does not match frozen contexts: {record_id}")
        check_raw_prediction(row)
        predictions[record_id] = row
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists():
        return predictions, None
    summary = json.loads(summary_path.read_text())
    data_hash = file_hash(data_path)
    ids_hash = file_hash(ids_path) if ids_path else None
    if (summary.get("status") != "complete" or summary.get("n") != len(predictions)
            or summary.get("data_sha256") != data_hash or summary.get("test_ids_sha256") != ids_hash):
        raise ValueError(f"Prediction summary does not match data or frozen IDs: {summary_path}")
    if (summary.get("adapter") is not None) != (method == Method.ADAPTED):
        raise ValueError(f"Adapter metadata does not match {method}: {summary_path}")
    return predictions, summary


def compare_metadata(metadata: dict[Method, Record | None]) -> None:
    available = [(method, metadata.get(method)) for method in (Method.BASE, Method.ADAPTED, Method.IN_CONTEXT)]
    available = [(method, value) for method, value in available if value is not None]
    if len(available) < 2:
        return
    reference_method, reference = available[0]
    assert reference is not None
    for method, candidate in available[1:]:
        assert candidate is not None
        for key in ("model", "revision", "seed", "decoding", "enable_thinking", "max_new_tokens"):
            if key not in reference or reference[key] != candidate.get(key):
                raise ValueError(f"Inference settings disagree for {reference_method} and {method}: {key}")
    for key in ("dtype", "prompt_sha256"):
        present = [(method, value[key]) for method, value in available if value is not None and key in value]
        if present and any(value != present[0][1] for _, value in present[1:]):
            raise ValueError(f"Inference settings disagree for {key}: {[method for method, _ in present]}")


def missing_comparison_metadata(metadata: dict[Method, Record | None]) -> dict[Method, list[str]]:
    """Record historical settings that were not captured rather than guessing."""
    fields = ("dtype", "prompt_sha256")
    return {method: [field for field in fields if value is None or field not in value]
            for method, value in metadata.items()}


def bootstrap(values: list[float], rng: random.Random, count: int = 2000) -> list[float] | None:
    if len(values) < 5:
        return None
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(count))
    return [draws[int(count * 0.025)], draws[min(count - 1, int(count * 0.975))]]


def board_means(rows: list[Record], value: Any) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["layout_id"], []).append(value(row))
    return [sum(values) / len(values) for _, values in sorted(grouped.items())]


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def recover_progress(
    path: Path,
    config: Record,
    contexts: list[Record],
    resume: bool,
    config_name: str = "assessment_config.json",
) -> list[Record]:
    """Keep each completed measurement and validate its provenance on resume."""
    config_path = path.with_name(config_name)
    if path.exists() and path.stat().st_size and not resume:
        raise ValueError(f"{path} already contains results; use --resume or a new output directory")
    if (config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config):
        raise ValueError("Assessment configuration or input hashes changed; use a new output directory")
    if path.exists() and path.stat().st_size and not config_path.exists():
        raise ValueError("Cannot resume without the original assessment configuration")
    atomic_write(config_path, json.dumps(config, indent=2) + "\n")
    if not path.exists():
        path.touch()
        return []
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        # Preserve the exact interrupted file before repairing a partial line.
        backup = path.with_name(f"{path.name}.interrupted-{time.time_ns()}")
        backup.write_bytes(content)
        previous, separator, tail = content.rpartition(b"\n")
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            content = previous + separator
        else:
            content += b"\n"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    expected = {record["record_id"]: record["layout_id"] for record in contexts}
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        key = row.get("method"), row.get("record_id")
        if (key in seen or row.get("method") not in set(Method) or row.get("record_id") not in expected
                or row.get("layout_id") != expected[row["record_id"]]
                or row.get("status") not in set(AssessmentStatus)):
            raise ValueError(f"Invalid saved assessment record: {key}")
        seen.add(key)
    return rows


def assess_context(
    record: Record,
    method: Method,
    predictions: dict[Method, dict[str, Record]],
    max_depth: int,
    max_nodes: int,
    seconds: float,
) -> Record:
    prediction: Record | None
    if method == Method.LOCAL_RULE:
        local_intent = local_rule(record["start_observation"])
        prediction = {
            "parsed": local_intent,
            "raw_output": json.dumps(local_intent),
            "json_valid": True,
        }
    else:
        prediction = predictions[method].get(record["record_id"])
    row = {
        "record_id": record["record_id"],
        "layout_id": record["layout_id"],
        "method": method,
    }
    if prediction is None:
        row.update({
            "status": AssessmentStatus.MISSING,
            "json_valid": None,
            "valid_nontrivial": None,
            "solver": None,
        })
        return row
    intent = prediction.get("parsed")
    errors = (validate_prediction(intent, record["start_observation"])
              if prediction["json_valid"] else ["invalid JSON"])
    valid = not errors
    solver = None
    if valid:
        if not isinstance(intent, dict):
            raise ValueError("A valid intent must be an object")
        solver = bounded_witness(record["start_observation"], intent, max_depth, max_nodes, seconds)
    effect_value = intent.get("effects") if isinstance(intent, dict) else None
    effects = effect_value if isinstance(effect_value, list) else []
    row.update({
        "status": AssessmentStatus.EVALUATED,
        "json_valid": prediction["json_valid"],
        "valid_nontrivial": valid,
        "validation_errors": errors,
        "intent": intent,
        "raw_output": prediction.get("raw_output"),
        "effect_types": [effect.get("predicate") for effect in effects if isinstance(effect, dict)],
        "conjunction_size": len(effects),
        "solver": solver,
        "generation_seconds": prediction.get("generation_seconds"),
        "start_observation": record["start_observation"],
        "demonstration_later_observation": record["later_observation"],
    })
    return row


def has_witness(row: Record) -> bool:
    return bool(row.get("solver") and row["solver"]["witness_found"])


def paired_comparison(results: list[Record], reference: Method, candidate: Method) -> Record:
    """Compare the same contexts and bootstrap their independent boards."""
    by_method = {
        method: {
            row["record_id"]: row
            for row in results
            if row["method"] == method and row["status"] == AssessmentStatus.EVALUATED
        } for method in (reference, candidate)
    }
    paired = []
    for record_id in sorted(by_method[reference].keys() & by_method[candidate].keys()):
        before, after = by_method[reference][record_id], by_method[candidate][record_id]
        if before["layout_id"] != after["layout_id"]:
            raise ValueError(f"Paired context has inconsistent layouts: {record_id}")
        paired.append({
            "layout_id": before["layout_id"],
            "valid_delta": int(after["valid_nontrivial"]) - int(before["valid_nontrivial"]),
            "witness_delta": int(has_witness(after)) - int(has_witness(before)),
        })
    comparison: Record = {
        "contexts": len(paired),
        "boards": len({row["layout_id"] for row in paired}),
    }
    for metric in ("valid_delta", "witness_delta"):
        values = board_means(paired, lambda row: row[metric])
        comparison[metric] = sum(values) / len(values) if values else None
        comparison[f"{metric}_95_board_bootstrap"] = bootstrap(values, random.Random(20261114))
    return comparison


def summarise(results: list[Record]) -> tuple[list[Record], Record]:
    summary = []
    methods = [Method.BASE, Method.ADAPTED]
    if any(row["method"] == Method.IN_CONTEXT for row in results):
        methods.append(Method.IN_CONTEXT)
    methods.append(Method.LOCAL_RULE)
    for method in methods:
        planned = [row for row in results if row["method"] == method]
        available = [row for row in planned if row["status"] == AssessmentStatus.EVALUATED]
        good = [row for row in available if row["valid_nontrivial"]]
        witness = [row for row in good if has_witness(row)]
        valid_boards = board_means(available, lambda row: int(row["valid_nontrivial"]))
        witness_boards = board_means(available, lambda row: int(has_witness(row)))
        rng = random.Random(20261114)
        summary.append({
            "method": method,
            "planned_contexts": len(planned),
            "planned_boards": len({row["layout_id"] for row in planned}),
            "outputs": len(available),
            "boards_with_outputs": len(valid_boards),
            "missing_infrastructure_outputs": len(planned) - len(available),
            "valid_json": sum(row["json_valid"] for row in available),
            "valid_nontrivial": len(good),
            "witness_found": len(witness),
            "unresolved_among_valid": len(good) - len(witness),
            "valid_json_fraction": sum(row["json_valid"] for row in available) / len(available) if available else None,
            "valid_fraction": len(good) / len(available) if available else None,
            "witness_fraction_all_outputs": len(witness) / len(available) if available else None,
            "witness_fraction_valid": len(witness) / len(good) if good else None,
            "board_mean_valid_fraction": sum(valid_boards) / len(valid_boards) if valid_boards else None,
            "board_mean_witness_fraction": sum(witness_boards) / len(witness_boards) if witness_boards else None,
            "valid_fraction_95_board_bootstrap": bootstrap(valid_boards, rng),
            "witness_fraction_95_board_bootstrap": bootstrap(witness_boards, rng),
            "effect_types": dict(Counter(effect for row in good for effect in row["effect_types"])),
            "conjunction_sizes": dict(Counter(str(row["conjunction_size"]) for row in good)),
            "solver_statuses": dict(Counter(row["solver"]["status"] for row in good)),
            "solver_expanded_nodes": sum(row["solver"]["expanded_nodes"] for row in good),
            "solver_seconds": sum(row["solver"]["elapsed_seconds"] for row in good),
        })
    return summary, paired_comparison(results, Method.BASE, Method.ADAPTED)


def write_examples(results: list[Record], output: Path) -> None:
    """Record representative successes and failures without changing metrics."""
    selected: list[Record] = []
    for method in Method:
        eligible = [row for row in results if row["method"] == method and row["status"] == AssessmentStatus.EVALUATED]
        for matches in (
                lambda row: not row["valid_nontrivial"],
                lambda row: row["valid_nontrivial"] and has_witness(row),
                lambda row: row["valid_nontrivial"] and not has_witness(row),
        ):
            example = next((row for row in eligible if matches(row)), None)
            if example is not None:
                selected.append(example)
    examples = []
    for row in selected:
        solver = row["solver"]
        witness = solver.get("reached_observation") if solver else None
        predicates = []
        if row["valid_nontrivial"]:
            for group in ("effects", "preserve"):
                for predicate in row["intent"].get(group, []):
                    predicates.append({
                        "group": group,
                        "predicate": predicate,
                        "start": predicate_value(predicate, row["start_observation"]),
                        "demonstration_end": predicate_value(predicate, row["demonstration_later_observation"]),
                        "solver_end": predicate_value(predicate, witness) if witness else None,
                    })
        examples.append({
            "method": row["method"],
            "record_id": row["record_id"],
            "layout_id": row["layout_id"],
            "initial_board": row["start_observation"],
            "raw_output": row["raw_output"],
            "parsed_intent": row["intent"],
            "validation_errors": row["validation_errors"],
            "predicate_truth": predicates,
            "solver": solver,
            "demonstration_end_board": row["demonstration_later_observation"],
        })
    atomic_write(
        output / "examples.json",
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
    )
    lines = [
        "# Selected held-out examples",
        "",
        "Illustrative examples are selected by method and outcome; all metrics use the full frozen subset.",
        "The demonstration end is shown for context and was not supplied to the generator. "
        "Any reached state is a bounded-search witness replayed in the environment, not a controller rollout.",
    ]
    for example in examples:
        solver = example["solver"]
        lines.extend([
            "",
            f"## {example['method']} — {example['record_id']}",
            "",
            "Current board:",
            "",
            "```text",
            *example["initial_board"],
            "```",
            "",
            "Raw output:",
            "",
            "````text",
            example["raw_output"],
            "````",
            "",
            "Parsed intent and validation:",
            "",
            "```json",
            json.dumps(
                {
                    "intent": example["parsed_intent"],
                    "errors": example["validation_errors"],
                    "predicate_truth": example["predicate_truth"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "Recorded demonstration end:",
            "",
            "```text",
            *example["demonstration_end_board"],
            "```",
        ])
        if solver and solver["witness_found"]:
            lines.extend([
                "",
                f"Bounded-search witness: {solver['witness_depth']} primitive actions.",
                "",
                "```text",
                *solver["reached_observation"],
                "```",
            ])
        elif solver:
            lines.extend([
                "",
                f"Bounded search: {solver['status']}; no witness found within this search.",
            ])
        else:
            lines.extend(["", "Bounded search was not run for this invalid intent."])
    atomic_write(output / "examples.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "assess"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--adapted", type=Path)
    parser.add_argument("--in-context", type=Path)
    parser.add_argument("--test-ids", type=Path)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_depth < 0 or args.max_nodes < 1 or args.seconds <= 0:
        parser.error("search depth must be nonnegative and node/time limits positive")
    if args.test_ids:
        expected_hash = json.loads(args.test_ids.read_text(encoding="utf-8")).get("dataset_sha256")
        if expected_hash is not None and expected_hash != file_hash(args.data):
            raise ValueError("Frozen contexts belong to a different dataset")
    contexts = test_contexts(read_jsonl(args.data), args.test_ids)
    if not contexts:
        raise ValueError("No held-out contexts")
    if args.mode == "freeze":
        result = {
            "selection": "all preselected checked test records, ordered by frozen dataset",
            "dataset_sha256": file_hash(args.data),
            "record_ids": [r["record_id"] for r in contexts],
            "layout_ids": sorted({r["layout_id"] for r in contexts}),
            "count": len(contexts),
            "independent_boards": len({r["layout_id"] for r in contexts}),
        }
        atomic_write(args.output, json.dumps(result, indent=2) + "\n")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    predictions: dict[Method, dict[str, Record]] = {}
    metadata: dict[Method, Record | None] = {}
    inputs = [(Method.BASE, args.base), (Method.ADAPTED, args.adapted)]
    if args.in_context is not None:
        inputs.append((Method.IN_CONTEXT, args.in_context))
    for method, path in inputs:
        predictions[method], metadata[method] = predictions_for(path, method, contexts, args.data, args.test_ids)
    compare_metadata(metadata)
    config = {
        "schema_version": 1,
        "data_sha256": file_hash(args.data),
        "test_ids_sha256": file_hash(args.test_ids) if args.test_ids else None,
        "prediction_sha256": {method: file_hash(path) if path and path.exists() else None for method, path in inputs},
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "seconds": args.seconds,
    }
    output_jsonl = args.output / "assessment.jsonl"
    results = recover_progress(output_jsonl, config, contexts, args.resume)
    seen = {(row["method"], row["record_id"]) for row in results}
    with output_jsonl.open("a", encoding="utf-8", buffering=1) as stream:
        methods = [method for method, _ in inputs] + [Method.LOCAL_RULE]
        for method in methods:
            for record in contexts:
                if (method, record["record_id"]) in seen:
                    continue
                row = assess_context(
                    record,
                    method,
                    predictions,
                    args.max_depth,
                    args.max_nodes,
                    args.seconds,
                )
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    json.dumps({
                        "method": method,
                        "record_id": record["record_id"],
                        "valid": row["valid_nontrivial"],
                        "solver": row["solver"]["status"] if row["solver"] else None,
                    }),
                    flush=True,
                )
    summary, comparison = summarise(results)
    result = {
        "results": summary,
        "paired_adapted_minus_base": comparison,
        "prediction_metadata": metadata,
        "solver": {
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "seconds": args.seconds,
            "meaning": "bounded solver witness; not controller reachability",
        },
    }
    if args.in_context is not None:
        result["paired_in_context_minus_base"] = paired_comparison(results, Method.BASE, Method.IN_CONTEXT)
        result["paired_in_context_minus_adapted"] = paired_comparison(results, Method.ADAPTED, Method.IN_CONTEXT)
        result["comparison_metadata_missing"] = missing_comparison_metadata(metadata)
    write_examples(results, args.output)
    columns = [
        "method",
        "planned_contexts",
        "planned_boards",
        "outputs",
        "boards_with_outputs",
        "missing_infrastructure_outputs",
        "valid_json",
        "valid_nontrivial",
        "witness_found",
        "unresolved_among_valid",
        "valid_json_fraction",
        "valid_fraction",
        "witness_fraction_all_outputs",
        "witness_fraction_valid",
        "board_mean_valid_fraction",
        "board_mean_witness_fraction",
        "solver_expanded_nodes",
        "solver_seconds",
    ]
    output_csv = args.output / "summary.csv"
    temporary = output_csv.with_name(f".{output_csv.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    os.replace(temporary, output_csv)
    # The summary is the completion marker and must be written last.
    atomic_write(args.output / "summary.json", json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
