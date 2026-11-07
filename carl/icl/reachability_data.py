"""Offline, replay-verified data freeze for trajectory-target reachability.

Evaluator artifacts are not public initial-state manifests. Never pass them or
witness actions to a policy: construct each query from the selected two boards.
"""
import contextlib
import gzip
import hashlib
import io
import json
import sys
import tarfile
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.boards import canonical_layout_id, from_rows, level_key, state_hash, to_rows
from carl.icl.data_sources.sokoban import environment, verify_solution
from carl.icl.luna_components.formatting import task

VERSION = "reachability-data-v2"
FAMILY = "real_12x12x4"
OFFSETS = (1, 2, 4, 8, 12, 16)
NAMESPACES = frozenset({"pilot", "main", "calibration_A", "calibration_B"})
ARMS = ("luna", "terra_6_1")
SOURCE_FILES = (
    "training.json", "value_examples.json", "fresh_holdout_manifest.json",
    f"{FAMILY}/validation.json", f"{FAMILY}/test.json",
    f"private_witnesses/{FAMILY}/validation.json", f"private_witnesses/{FAMILY}/test.json",
)
RUNTIME_FILES = (
    "carl/icl/reachability_data.py", "carl/icl/reachability_history.py", "carl/icl/boards.py",
    "carl/icl/data_sources/sokoban.py", "carl/icl/luna_components/formatting.py", "carl/icl/types.py",
    "carl/environment/sokoban/env.py", "carl/environment/sokoban/core.py", "carl/environment/sokoban/tokenizer.py",
    "scripts/figure1_reachability.py", "uv.lock",
)


def canonical_bytes(value: Any) -> bytes:
    """Stable JSON bytes independent of indentation and insertion order."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(digest(canonical_bytes([VERSION, *parts]))[:16], 16)


def trial_seed(master_seed: int, namespace: str, pair_id: str, trial: int) -> int:
    """Use disjoint domains without depending on process or iteration order."""
    if namespace not in NAMESPACES or not pair_id or trial < 0 or master_seed < 0:
        raise ValueError("Invalid trial seed inputs")
    return stable_seed("trial", master_seed, namespace, pair_id, trial)


def eligible_anchors(states: list[np.ndarray], offsets: tuple[int, ...]) -> tuple[list[int], dict[str, int]]:
    """Keep all K at one anchor, excluding source-equal targets only."""
    if not offsets or any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("Offsets must be distinct positive integers")
    anchors = []
    counts = {"total": max(len(states) - 1, 0), "insufficient_future": 0, "source_equals_target": 0, "eligible": 0}
    for i in range(len(states) - 1):
        if i + max(offsets) >= len(states):
            counts["insufficient_future"] += 1
        elif any(state_hash(states[i]) == state_hash(states[i + k]) for k in offsets):
            counts["source_equals_target"] += 1
        else:
            anchors.append(i)
            counts["eligible"] += 1
    return anchors, counts


def sample_reference_targets(states: list[np.ndarray], position: int, *, window: int, count: int, seed: int) -> list[int]:
    """Sample uniformly over distinct non-source future states, not offsets."""
    if position < 0 or position >= len(states) or window < 1 or count < 1 or seed < 0:
        raise ValueError("Invalid reference sampling input")
    seen = {state_hash(states[position])}
    offsets = []
    for k in range(1, min(window, len(states) - position - 1) + 1):
        target = state_hash(states[position + k])
        if target not in seen:
            offsets.append(k)
            seen.add(target)
    if not offsets:
        return []
    selected = np.random.default_rng(seed).choice(offsets, min(count, len(offsets)), replace=False)
    return sorted(int(k) for k in selected)


def validate_split_disjoint(groups: dict[str, list[np.ndarray]]) -> None:
    """Reject D4-equivalent layouts across splits, not repetitions inside a split."""
    seen: dict[str, str] = {}
    for name, boards in groups.items():
        for board in boards:
            layout = canonical_layout_id(board)
            if layout in seen and seen[layout] != name:
                raise ValueError(f"Canonical layout overlaps splits {seen[layout]} and {name}")
            seen[layout] = name


def _json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


def _action_witness(initial: np.ndarray, actions: list[int]) -> list[np.ndarray]:
    if not isinstance(actions, list) or any(not isinstance(action, int) or isinstance(action, bool) or action not in range(4) for action in actions):
        raise ValueError("Witness actions must be integer UP=0, DOWN=1, LEFT=2, RIGHT=3")
    with contextlib.redirect_stdout(io.StringIO()):
        states = verify_solution(initial, actions)
    return states


def _training_check(training: dict) -> list[np.ndarray]:
    boards = []
    for context in training["contexts"]:
        board = from_rows(context["state"])
        boards.append(board)
        with contextlib.redirect_stdout(io.StringIO()):
            env = environment(board)
        for candidate in context["candidates"]:
            actions = candidate["actions"]
            if (not isinstance(actions, list) or any(not isinstance(a, int) or isinstance(a, bool) or a not in range(4) for a in actions)
                    or len(actions) != candidate["k"]):
                raise ValueError("Invalid training demonstration actions/offset")
            current = board.copy()
            for action in actions:
                if env.is_solved(current):
                    raise ValueError("Training demonstration continues after terminal")
                nxt = np.asarray(env.next_state(current, action))
                if np.array_equal(current, nxt):
                    raise ValueError("Training demonstration contains a no-op")
                current = nxt
            if not np.array_equal(current, from_rows(candidate["subgoal"])):
                raise ValueError("Training demonstration target does not replay")
    return boards


def _records(public: dict, private: list, split: str) -> tuple[list[dict], list[dict]]:
    rows = public["instances"]
    witnesses = {r["id"]: r["actions"] for r in private}
    if len(witnesses) != len(private) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {split}")
    if {r["id"] for r in rows} != set(witnesses):
        raise ValueError(f"Public/witness IDs mismatch in {split}")
    result, errors = [], []
    for row in rows:
        try:
            state = from_rows(row["state"])
            if state.shape[:2] != (12, 12) or int(np.isin(state.argmax(-1), [3, 4]).sum()) != 4:
                raise ValueError("Unexpected board family")
            if "level_key" in row and row["level_key"] != level_key(state):
                raise ValueError("Source board layout checksum mismatch")
            if "state_hash" in row and row["state_hash"] != state_hash(state):
                raise ValueError("Source board state checksum mismatch")
            if "layout_id" in row and row["layout_id"] != canonical_layout_id(state):
                raise ValueError("Source canonical layout checksum mismatch")
            states = _action_witness(state, witnesses[row["id"]])
            anchors, counts = eligible_anchors(states, OFFSETS)
            result.append({"trajectory_id": row["id"], "layout_id": canonical_layout_id(state),
                "actions": witnesses[row["id"]], "states": states, "anchors": anchors, "counts": counts})
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            errors.append({"trajectory_id": row.get("id"), "reason": str(exc)})
    return result, errors


def _bundle(files: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(files.items()):
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                entry.mode = 0o600
                entry.uid = entry.gid = entry.mtime = 0
                entry.uname = entry.gname = ""
                archive.addfile(entry, io.BytesIO(data))
    return raw.getvalue()


def _unbundle(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        result = {}
        for member in archive.getmembers():
            if not member.isfile() or member.name in result or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("Invalid bundle entry")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Unreadable bundle entry")
            result[member.name] = handle.read()
    return result


def _identifier(kind: str, *parts: Any) -> str:
    return f"{kind}-" + digest(canonical_bytes([VERSION, kind, *parts]))[:24]


def _trial_ids(master_seed: int, namespace: str, pair_id: str, repetitions: int) -> dict[str, list[dict]]:
    # Pair RNG streams across arms, while preserving distinct result identities.
    return {arm: [{"trial": i, "trial_id": _identifier("trial", master_seed, namespace, pair_id, arm, i),
                   "action_seed": trial_seed(master_seed, namespace, pair_id, i)} for i in range(repetitions)]
            for arm in ARMS}


def _check_fragment(states: list[np.ndarray], actions: list[int], i: int, k: int) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        env = environment(states[i])
    current = states[i].copy()
    for action in actions[i:i + k]:
        current = np.asarray(env.next_state(current, action))
    if state_hash(current) != state_hash(states[i + k]):
        raise ValueError("Fragment replay endpoint mismatch")


def _source_shard_evidence(repo: Path, files: dict[str, bytes]) -> dict:
    """Check witness states against the original shard when it is available."""
    expected = _json(files["source/fresh_holdout_manifest.json"])["source_sha256"]
    relative = "rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_1.pkl"
    source = repo / relative
    if not source.is_file():
        return {"status": "unavailable", "path": relative, "sha256_expected": expected,
                "note": "The bundle still contains full initial states and action witnesses."}
    observed = digest(source.read_bytes())
    if observed != expected:
        raise ValueError("Original trajectory shard SHA-256 differs from provenance")
    # This is an existing, explicitly supplied local joblib source, never downloaded.
    import joblib

    raw = joblib.load(source)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    checked, failures = 0, []
    for split in ("validation", "test"):
        records = _json(files[f"source/{FAMILY}/{split}.json"])["instances"]
        witnesses = {r["id"]: r["actions"] for r in _json(files[f"source/private_witnesses/{FAMILY}/{split}.json"])}
        for row in records:
            identifier = row["id"]
            try:
                if not identifier.startswith("trajectory_"):
                    raise ValueError("Unknown source trajectory identifier")
                index = int(identifier.removeprefix("trajectory_"))
                if index < 0 or index >= len(trajectories):
                    raise ValueError("Source trajectory index outside shard")
                states = _action_witness(from_rows(row["state"]), witnesses[identifier])
                original = np.asarray(trajectories[index])
                if len(original) < len(states) or any(not np.array_equal(a, b) for a, b in zip(states, original, strict=False)):
                    raise ValueError("Replayed witness differs from source trajectory")
                checked += 1
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                failures.append({"trajectory_id": identifier, "split": split, "reason": str(exc)})
    return {"status": "checked" if not failures else "partial", "path": relative, "sha256": observed,
            "source_trajectory_count": len(trajectories), "verified_witness_prefixes": checked,
            "failures": failures, "note": "Source suffix after the solved witness is not used."}


def policy_query(pair: dict, state_store: dict[str, list[str]]) -> dict[str, list[str]]:
    """Expose only the selected source and target, never evaluator metadata."""
    source, target = state_store[pair["source_hash"]], state_store[pair["target_hash"]]
    if state_hash(from_rows(source)) != pair["source_hash"] or state_hash(from_rows(target)) != pair["target_hash"]:
        raise ValueError("Policy query state hash mismatch")
    return {"state": source, "subgoal": target}


def _build(files: dict[str, bytes], *, seed: int, n_main: int) -> dict[str, bytes]:
    if seed < 0 or not 1 <= n_main <= 100:
        raise ValueError("seed must be nonnegative and n_main must be in 1..100")
    training = _json(files["source/training.json"])
    provenance = _json(files["source/fresh_holdout_manifest.json"])
    if training.get("provenance", {}).get("source_sha256") != provenance.get("source_sha256"):
        raise ValueError("Training and selected trajectories name different source shards")
    public_ids = [r["id"] for r in _json(files[f"source/{FAMILY}/test.json"])["instances"]]
    if set(public_ids) != set(provenance["ids"]) or len(public_ids) != len(provenance["ids"]):
        raise ValueError("Test trajectory IDs do not match their provenance manifest")
    groups, errors = {}, {}
    for split, source_split in (("pilot", "validation"), ("main_exploratory", "test")):
        groups[split], errors[split] = _records(_json(files[f"source/{FAMILY}/{source_split}.json"]),
            _json(files[f"source/private_witnesses/{FAMILY}/{source_split}.json"]), split)
    source_evidence = _json(files["history/source_shard_evidence.json"])
    for split, source_split in (("pilot", "validation"), ("main_exploratory", "test")):
        failures = {row["trajectory_id"]: row["reason"] for row in source_evidence.get("failures", [])
                    if row["split"] == source_split}
        kept = []
        for record in groups[split]:
            if record["trajectory_id"] in failures:
                errors[split].append({"trajectory_id": record["trajectory_id"],
                                      "reason": "source_shard: " + failures[record["trajectory_id"]]})
            else:
                kept.append(record)
        groups[split] = kept
    training_boards = _training_check(training)
    if len({canonical_layout_id(board) for board in training_boards}) != len(training_boards):
        raise ValueError("Training demonstrations contain duplicate D4 layouts")
    validate_split_disjoint({"training": training_boards, **{s: [r["states"][0] for r in rows]
                                                              for s, rows in groups.items()}})
    states_store: dict[str, list[str]] = {}
    core, selected = [], {}
    reports: dict[str, Any] = {}
    for split, records in groups.items():
        by_layout: dict[str, list[dict]] = defaultdict(list)
        rejected = []
        for record in records:
            if record["anchors"]:
                by_layout[record["layout_id"]].append(record)
            else:
                rejected.append({"trajectory_id": record["trajectory_id"], "layout_id": record["layout_id"],
                                 "reason": "no_eligible_anchor"})
        layouts = sorted(by_layout)
        layout_seed = stable_seed(seed, "layout-selection", split)
        perm = np.random.default_rng(layout_seed).permutation(len(layouts))
        limit = n_main if split == "main_exploratory" else len(layouts)
        order = [layouts[int(j)] for j in perm[:limit]]
        chosen = []
        for rank, layout in enumerate(order):
            trajectories = sorted(by_layout[layout], key=lambda x: x["trajectory_id"])
            trajectory_seed = stable_seed(seed, "trajectory", split, layout)
            rng = np.random.default_rng(trajectory_seed)
            record = trajectories[int(rng.integers(len(trajectories)))]
            anchor_seed = stable_seed(seed, "anchor", split, layout, record["trajectory_id"])
            rng = np.random.default_rng(anchor_seed)
            i = record["anchors"][int(rng.integers(len(record["anchors"])))]
            context_id = _identifier("context", split, layout, record["trajectory_id"], i)
            source_hash = state_hash(record["states"][i])
            states_store[source_hash] = to_rows(record["states"][i])
            witness_file = f"source/private_witnesses/{FAMILY}/{'test' if split == 'main_exploratory' else 'validation'}.json"
            context = {"split": split, "family": FAMILY, "rank": rank, "layout_id": layout,
                "trajectory_id": record["trajectory_id"], "context_id": context_id, "i": i,
                "trajectory_selection_seed": trajectory_seed, "anchor_selection_seed": anchor_seed,
                "trajectory_actions": len(record["actions"]), "source_hash": source_hash,
                "goal": task(record["states"][i]), "witness_sha256": digest(canonical_bytes(record["actions"])),
                "witness_bundle_entry": witness_file, "witness_file_sha256": digest(files[witness_file]),
                "trajectory_source_sha256": _json(files["source/fresh_holdout_manifest.json"])["source_sha256"]}
            chosen.append((record, context))
            for k in OFFSETS:
                _check_fragment(record["states"], record["actions"], i, k)
                target = record["states"][i + k]
                target_hash = state_hash(target)
                states_store[target_hash] = to_rows(target)
                pair_id = _identifier("pair", context_id, k, target_hash)
                core.append({**context, "k": k, "target_hash": target_hash, "pair_id": pair_id,
                    "alias_group": _identifier("alias", context_id, target_hash),
                    "trials": _trial_ids(seed, "pilot" if split == "pilot" else "main", pair_id, 5)})
        selected[split] = chosen
        reports[split] = {"input_trajectories": len(records) + len(errors[split]),
            "verified_trajectories": len(records), "invalid_witnesses": errors[split],
            "distinct_layouts": len({r["layout_id"] for r in records}), "eligible_layouts": len(layouts),
            "rejected_layouts": sorted({r["layout_id"] for r in records} - set(layouts)),
            "requested_layouts": limit, "selected_layouts": len(chosen), "excluded_trajectories": rejected,
            "layout_selection_seed": layout_seed,
            "anchor_counts": {key: sum(r["counts"][key] for r in records)
                              for key in ("total", "insufficient_future", "source_equals_target", "eligible")},
            "trajectory_lengths_actions": sorted(len(r["actions"]) for r in records),
            "reduction": max(0, limit - len(chosen))}
    main = selected["main_exploratory"]
    calibration = []
    calibration_context_seed = stable_seed(seed, "calibration-contexts")
    # Choose from the fixed N50 prefix, so the same diagnostic is available
    # when running either N50 or N100 from this frozen cohort.
    calibration_population = main[:50]
    calibration_order = np.random.default_rng(calibration_context_seed).permutation(len(calibration_population))
    for ci in calibration_order[:min(8, len(main))]:
        record, context = calibration_population[int(ci)]
        i = context["i"]
        target_seed = stable_seed(seed, "calibration-targets", context["context_id"])
        offsets = sample_reference_targets(record["states"], i, window=16, count=8, seed=target_seed)
        for k in offsets:
            _check_fragment(record["states"], record["actions"], i, k)
            target_hash = state_hash(record["states"][i + k])
            states_store[target_hash] = to_rows(record["states"][i + k])
            aliases = [j for j in range(1, 17) if state_hash(record["states"][i + j]) == target_hash]
            pair_id = _identifier("reference", context["context_id"], target_hash)
            calibration.append({**context, "k": k, "window_aliases": aliases,
                "target_selection_seed": target_seed,
                "target_hash": target_hash, "pair_id": pair_id,
                "groups": {label: _trial_ids(seed, f"calibration_{label}", pair_id, 10) for label in ("A", "B")}})
    history = _json(files["history/scan.json"])
    for split, chosen in selected.items():
        ids = {ctx["layout_id"] for _, ctx in chosen}
        for name, item in history["sources"].items():
            source_path = item.get("path", "")
            training_source = (source_path.endswith("training.json") or "demonstrations.json" in source_path
                               or source_path.endswith("assets/checkpoint/generator.json")
                               or source_path.endswith("assets/verbal.json"))
            if item.get("status") == "available" and training_source and ids.intersection(item.get("layout_ids", [])):
                raise ValueError(f"Evaluation layouts overlap historical demonstrations: {name}")
        reports[split]["history_intersections"] = {
            name: len(ids.intersection(item.get("layout_ids", [])))
            for name, item in history["sources"].items() if item.get("status") == "available"}
    experiment_id = _identifier("experiment", seed, n_main, {name: digest(data) for name, data in sorted(files.items())})
    core_doc = {"schema": VERSION, "experiment": "icl-reachability-exploratory-v2", "experiment_id": experiment_id, "kind": "evaluator_pairs", "offset_unit": "primitive_actions",
                "models": list(ARMS), "pairs": core}
    ref_doc = {"schema": VERSION, "experiment": "icl-reachability-exploratory-v2", "experiment_id": experiment_id, "kind": "evaluator_reference_pairs", "models": list(ARMS), "window": 16,
               "max_distinct_targets": 8, "groups": {"A": 10, "B": 10}, "pairs": calibration}
    bundle = _bundle(files)
    derived = {"core_pairs.json": canonical_bytes(core_doc), "calibration_pairs.json": canonical_bytes(ref_doc),
               "states.json": canonical_bytes({"schema": VERSION, "states": states_store}),
               "private/source_bundle.tar.gz": bundle}
    manifest = {"schema": VERSION, "experiment": "icl-reachability-exploratory-v2", "experiment_id": experiment_id, "seed": seed,
        "family": FAMILY, "classification": "exploratory; not a virgin final test", "offsets": list(OFFSETS),
        "main_requested": n_main, "main_selected": len(main), "n50": [c["context_id"] for _, c in main[:50]],
        "n100": [c["context_id"] for _, c in main], "calibration_contexts": [calibration_population[int(i)][1]["context_id"]
            for i in calibration_order[:min(8, len(main))]],
        "n50_complete": len(main) >= 50, "n100_complete": len(main) >= 100,
        "calibration_context_selection_seed": calibration_context_seed,
        "calibration_context_population": "N50 prefix of the main exploratory cohort",
        "seed_contract": "SHA256 of canonical JSON [version, trial, master seed, namespace, pair ID, trial index]; first 64 bits",
        "sampling": {"layout": "uniform permutation of sorted eligible canonical IDs", "trajectory": "uniform within layout",
            "anchor": "uniform among full-window anchors with no source-equal K target",
            "reference": "uniform unique non-source state from full 1..16 window; first offset represents aliases",
            "rng": "numpy.default_rng(PCG64)", "numpy": np.__version__},
        "model_query_fields": ["state", "subgoal"], "reference_witnesses_are_not_optimal": True,
        "model_arms_are_labels_not_api_ids": True, "runtime": _json(files["runtime/metadata.json"]),
        "source_shard_evidence": source_evidence,
        "source_files": {name: digest(data) for name, data in sorted(files.items())},
        "artifacts_sha256": {name: digest(data) for name, data in sorted(derived.items())},
        "split_audit": reports, "historical_sources": {name: {k: v for k, v in entry.items() if k != "layout_ids"}
            for name, entry in history["sources"].items()},
        "coverage_limit": history["limitations"], "calibration_not_for_tuning": True}
    derived["manifest.json"] = canonical_bytes(manifest)
    report = {"schema": VERSION, "classification": manifest["classification"], "split_audit": reports,
              "core_pairs": len(core), "calibration_contexts": len(manifest["calibration_contexts"]),
              "core_pairs_by_split": {s: sum(row["split"] == s for row in core) for s in selected},
              "aliased_core_offsets_by_split": {s: sum(row["split"] == s for row in core)
                  - len({(row["context_id"], row["target_hash"]) for row in core if row["split"] == s})
                  for s in selected},
              "calibration_pairs": len(calibration), "calibration_groups": "A/B: 10 trials per target per model",
              "source_bundle_sha256": digest(bundle), "historical_sources": manifest["historical_sources"],
              "source_shard_evidence": source_evidence,
              "coverage_limit": manifest["coverage_limit"]}
    derived["dataset_report.json"] = canonical_bytes(report)
    text = ["# Reachability dataset report", "", f"- Classification: {manifest['classification']}",
            f"- Schema: {VERSION}; seed: {seed}; K: {list(OFFSETS)} primitive actions",
            f"- Main: {len(main)}/{n_main} layouts; N50 is a prefix: {len(main) >= 50}",
            f"- Pilot: {len(selected['pilot'])} layouts; total core pairs: {len(core)}",
            f"- Reference: {len(manifest['calibration_contexts'])} contexts, {len(calibration)} distinct pairs, A/B each 10 trials per arm",
            f"- Core pairs by split: {report['core_pairs_by_split']}",
            f"- Aliased non-source core offsets by split: {report['aliased_core_offsets_by_split']}",
            f"- Original source shard verification: {source_evidence['status']}; verified witness prefixes: {source_evidence.get('verified_witness_prefixes', 0)}",
            f"- Private source bundle SHA-256: {digest(bundle)}", ""]
    for split, summary in reports.items():
        text += [f"## {split}", "", f"- Input trajectories: {summary['input_trajectories']}",
                 f"- Verified trajectories: {summary['verified_trajectories']}",
                 f"- Invalid witnesses: {len(summary['invalid_witnesses'])}",
                 f"- Distinct/eligible/selected layouts: {summary['distinct_layouts']}/{summary['eligible_layouts']}/{summary['selected_layouts']}",
                 f"- Anchor counts: {summary['anchor_counts']}",
                 f"- Lengths in actions: {summary['trajectory_lengths_actions']}", ""]
    text += ["## Scope", "", "History overlap is not proof of model exposure. The full scan, missing sources and limitations are in the manifest and source bundle.",
             "Source/target boards and private witness actions must not be sent wholesale to policy. Model arms are labels, not verified API IDs.", ""]
    derived["dataset_report.md"] = ("\n".join(text) + "\n").encode()
    return derived


def prepare_dataset(source_root: Path, output: Path, *, seed: int = 20261103, n_main: int = 100) -> dict:
    """Build a new frozen bundle; refuse to overwrite any existing different data."""
    from carl.icl.reachability_history import scan_history

    files = {f"source/{name}": (source_root / name).read_bytes() for name in SOURCE_FILES}
    repo = Path(__file__).resolve().parents[2]
    files["history/scan.json"] = canonical_bytes(scan_history(repo))
    files["history/source_shard_evidence.json"] = canonical_bytes(_source_shard_evidence(repo, files))
    for name in RUNTIME_FILES:
        files[f"runtime/{name}"] = (repo / name).read_bytes()
    packages = {}
    for package in ("numpy", "joblib", "gym", "gymnasium"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    files["runtime/metadata.json"] = canonical_bytes({"python": sys.version.split()[0], "packages": packages,
        "code_sha256": {name: digest(files[f"runtime/{name}"]) for name in RUNTIME_FILES}})
    artifacts = _build(files, seed=seed, n_main=n_main)
    if output.exists():
        if not output.is_dir() or any(not (output / name).is_file() or (output / name).read_bytes() != data
                                      for name, data in artifacts.items()):
            raise ValueError("Existing freeze differs or is incomplete; choose a new version")
        return _json(artifacts["manifest.json"])
    output.mkdir(parents=True)
    try:
        for name, data in artifacts.items():
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name.startswith("private/"):
                destination.parent.chmod(0o700)
            destination.write_bytes(data)
            if name.startswith("private/"):
                destination.chmod(0o600)
    except Exception:
        # Do not silently reuse a partial output; a later invocation refuses it.
        raise
    return _json(artifacts["manifest.json"])


def verify_dataset(output: Path) -> dict:
    """Recreate all frozen bytes from the archived source and replay every fragment."""
    manifest = _json((output / "manifest.json").read_bytes())
    files = _unbundle((output / "private/source_bundle.tar.gz").read_bytes())
    runtime = _json(files["runtime/metadata.json"])
    repo = Path(__file__).resolve().parents[2]
    if any(not (repo / name).is_file() or digest((repo / name).read_bytes()) != expected
           for name, expected in runtime["code_sha256"].items()):
        raise ValueError("Runtime code or lockfile differs from archived preparation environment")
    if runtime["python"] != sys.version.split()[0] or runtime["packages"]["numpy"] != np.__version__:
        raise ValueError("Python or NumPy version differs from the archived preparation environment")
    artifacts = _build(files, seed=manifest["seed"], n_main=manifest["main_requested"])
    if any(not (output / name).is_file() or (output / name).read_bytes() != data for name, data in artifacts.items()):
        raise ValueError("Frozen artifact does not match replayed source bundle")
    return _json(artifacts["dataset_report.json"])
