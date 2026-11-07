"""Offline contracts for reproducible trajectory reachability datasets."""

from types import SimpleNamespace

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id, from_rows, state_hash
from carl.icl.data_sources.sokoban import environment, verify_solution
from carl.icl.reachability_data import (
    _trial_ids,
    eligible_anchors,
    policy_query,
    sample_reference_targets,
    trial_seed,
    validate_split_disjoint,
)
from carl.icl.search import rollout


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """Dataset preparation and witness validation must never contact a provider."""
    import socket

    def forbidden(*_args, **_kwargs):
        pytest.fail("Dataset preparation attempted network access")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def looped_trajectory():
    """A genuine Sokoban solution with source returns and repeated non-source targets."""
    initial = from_rows([
        "#######", "#-----#", "#-@---#", "#--$.-#", "#-----#", "#-----#", "#######",
    ])
    actions = [0, 1, 1, 0, 1, 3]  # UP, DOWN, DOWN, UP, DOWN, RIGHT
    states = verify_solution(initial, actions)
    return states, actions


def test_anchor_rejects_source_returns_but_preserves_target_aliases(looped_trajectory):
    states, _ = looped_trajectory
    offsets = (1, 2, 3, 4)
    anchors, counts = eligible_anchors(states, offsets)
    # At i=0 and i=2 the witness returns to source within K. At i=1,
    # k=1 and k=3 alias the same non-source board and must both be retained.
    assert anchors == [1]
    assert counts == {"total": 6, "insufficient_future": 3, "source_equals_target": 2, "eligible": 1}
    assert state_hash(states[2]) == state_hash(states[4])
    assert state_hash(states[1]) != state_hash(states[2])
    assert len([states[anchors[0] + k] for k in offsets]) == len(offsets)


def test_short_trajectory_is_reported_without_duplicate_sampling(looped_trajectory):
    states, _ = looped_trajectory
    anchors, counts = eligible_anchors(states, (1, 2, 4, 8, 12, 16))
    assert anchors == []
    assert counts["total"] == 6
    assert counts["insufficient_future"] == 6
    assert counts["eligible"] == 0


def test_reference_sampling_uses_full_window_distinct_states_and_is_seeded(looped_trajectory):
    states, _ = looped_trajectory
    # At source 0 there are only three non-source states in this window;
    # repeated visits are deduplicated before sampling, with earliest k retained.
    all_offsets = sample_reference_targets(states, 0, window=6, count=8, seed=91)
    assert set(all_offsets) == {1, 3, 6}
    assert len({state_hash(states[k]) for k in all_offsets}) == len(all_offsets)
    chosen = sample_reference_targets(states, 0, window=6, count=2, seed=91)
    assert chosen == sample_reference_targets(states, 0, window=6, count=2, seed=91)
    assert len(chosen) == 2
    assert set(chosen) <= {1, 3, 6}
    # Across independent seeds all three unique states receive sampling mass;
    # the loose bounds catch deterministic or duplicate-weighted selection.
    counts = {k: 0 for k in (1, 3, 6)}
    for seed in range(256):
        for k in sample_reference_targets(states, 0, window=6, count=2, seed=seed):
            counts[k] += 1
    assert all(110 <= count <= 230 for count in counts.values())


def test_d4_symmetries_cannot_cross_train_and_development_splits(looped_trajectory):
    states, _ = looped_trajectory
    board = states[0]
    rotated = np.rot90(board, axes=(0, 1)).copy()
    reflected = np.fliplr(board).copy()
    assert canonical_layout_id(board) == canonical_layout_id(rotated) == canonical_layout_id(reflected)
    for transformed in (rotated, reflected):
        with pytest.raises(ValueError):
            validate_split_disjoint({"training": [board], "development": [transformed]})


def test_distinct_layouts_can_be_assigned_to_disjoint_splits(looped_trajectory):
    states, _ = looped_trajectory
    other = from_rows([
        "#######", "#-----#", "#-@---#", "#--$.-#", "#-#---#", "#-----#", "#######",
    ])
    assert canonical_layout_id(states[0]) != canonical_layout_id(other)
    validate_split_disjoint({"training": [states[0]], "development": [other]})


def test_invalid_noop_and_unfinished_witnesses_cannot_be_replayed(looped_trajectory):
    states, _ = looped_trajectory
    initial = states[0]
    for actions in ([9], [2, 2], [1], [1, 3, 2]):
        with pytest.raises(ValueError):
            verify_solution(initial, actions)


@pytest.mark.parametrize("actions", [[9], [True], [1]])
def test_invalid_witness_is_listed_in_dataset_error_report(actions):
    from carl.icl.reachability_data import _records

    board = ["############", "#-@--------#", "#--$.$.$.$.#", *(["#----------#"] * 8), "############"]
    records, errors = _records({"instances": [{"id": "broken", "state": board}]},
                               [{"id": "broken", "actions": actions}], "main")
    assert records == []
    assert len(errors) == 1 and errors[0]["trajectory_id"] == "broken"
    assert errors[0]["reason"]


def test_all_selected_fragment_endpoints_replay_from_witness(looped_trajectory):
    states, actions = looped_trajectory
    anchors, _ = eligible_anchors(states, (1, 2, 3, 4))
    for i in anchors:
        for k in (1, 2, 3, 4):
            initial_hash = state_hash(states[i])
            current = states[i].copy()
            env = environment(current)
            for action in actions[i:i + k]:
                current = np.asarray(env.next_state(current, action))
            assert initial_hash == state_hash(states[i])
            assert state_hash(current) == state_hash(states[i + k])


def test_policy_query_contains_only_the_selected_boards_and_checks_hashes(looped_trajectory):
    from carl.icl.boards import to_rows

    states, _ = looped_trajectory
    source, target = states[1], states[2]
    source_hash, target_hash = state_hash(source), state_hash(target)
    store = {source_hash: to_rows(source), target_hash: to_rows(target)}
    pair = {"source_hash": source_hash, "target_hash": target_hash,
            "actions": [1, 3], "future_states": [to_rows(states[-1])], "witness": "private"}
    query = policy_query(pair, store)
    assert set(query) == {"state", "subgoal"}
    assert state_hash(from_rows(query["state"])) == source_hash
    assert state_hash(from_rows(query["subgoal"])) == target_hash
    corrupted = {**store, target_hash: to_rows(states[3])}
    with pytest.raises(ValueError, match="hash mismatch"):
        policy_query(pair, corrupted)


def test_initial_target_is_a_zero_action_hit_in_rollout(looped_trajectory):
    states, _ = looped_trajectory

    def forbidden(*_args):
        pytest.fail("A target already reached must not call the policy")

    result = rollout(environment(states[0]), SimpleNamespace(probabilities=forbidden),
                     states[0], states[0], cap=4, rng=np.random.default_rng(1))
    assert result.reached and len(result.actions) == 0


def test_trial_seeds_are_stable_and_separate_every_namespace():
    namespaces = ("pilot", "main", "calibration_A", "calibration_B")
    pairs = ("pair-a", "pair-b")
    jobs = [(namespace, pair, trial) for namespace in namespaces for pair in pairs for trial in range(10)]
    forward = {job: trial_seed(123, *job) for job in jobs}
    reversed_order = {job: trial_seed(123, *job) for job in reversed(jobs)}
    assert forward == reversed_order
    assert len(set(forward.values())) == len(jobs)
    assert all(seed != trial_seed(124, *job) for job, seed in forward.items())
    assert _trial_ids(123, "main", "pair-a", 1)["luna"][0]["trial_id"] != _trial_ids(124, "main", "pair-a", 1)["luna"][0]["trial_id"]


def test_unknown_trial_namespace_is_rejected():
    with pytest.raises(ValueError):
        trial_seed(123, "unexpected", "pair-a", 0)


def test_real_dataset_preparation_is_deterministic_and_never_overwrites(tmp_path):
    """The same actual witnesses must freeze identically in independent directories."""
    import hashlib
    import json
    from pathlib import Path

    from carl.icl.datasets import _public_instances
    from carl.icl.reachability_data import prepare_dataset, verify_dataset

    root = Path("data/icl/component_study_100_clean")
    if not (root / "private_witnesses/real_12x12x4/test.json").exists():
        pytest.skip("Real prepared Sokoban witnesses are not available locally")
    first, second, larger = tmp_path / "first", tmp_path / "second", tmp_path / "larger"
    prepare_dataset(root, first, seed=20261103, n_main=2)
    prepare_dataset(root, second, seed=20261103, n_main=2)
    names = ("manifest.json", "core_pairs.json", "calibration_pairs.json", "states.json")
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
        json.loads((first / name).read_text(encoding="utf-8"))
    prepare_dataset(root, larger, seed=20261103, n_main=3)
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    larger_manifest = json.loads((larger / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["experiment_id"] != larger_manifest["experiment_id"]
    assert first_manifest["n100"] == larger_manifest["n100"][:2]
    small_pairs = [r for r in json.loads((first / "core_pairs.json").read_text())["pairs"]
                   if r["split"] == "main_exploratory"]
    large_prefix = [r for r in json.loads((larger / "core_pairs.json").read_text())["pairs"]
                    if r["split"] == "main_exploratory" and r["rank"] < 2]
    assert small_pairs == large_prefix
    assert first_manifest["schema"] and first_manifest["experiment"]
    assert first_manifest["family"] == "real_12x12x4"
    assert first_manifest["main_selected"] == 2
    assert first_manifest["n50"] == first_manifest["n100"]
    assert first_manifest["source_files"]
    assert all(len(value) == 64 for value in first_manifest["source_files"].values())
    boards = json.loads((first / "states.json").read_text())["states"]
    for key, rows in boards.items():
        assert state_hash(from_rows(rows)) == key
    assert len(small_pairs) == 2 * 6
    # Evaluator metadata must not be accepted by the solver's public loader.
    improperly_public = tmp_path / "improper_public.json"
    improperly_public.write_text(json.dumps({"instances": [small_pairs[0]], "provenance": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        _public_instances(improperly_public)
    assert (first / "private/source_bundle.tar.gz").stat().st_mode & 0o077 == 0
    for context_id in first_manifest["n100"]:
        selected = [r for r in small_pairs if r["context_id"] == context_id]
        assert {r["k"] for r in selected} == {1, 2, 4, 8, 12, 16}
        assert len({r["source_hash"] for r in selected}) == 1
        assert len({r["i"] for r in selected}) == 1
        assert all(r["source_hash"] != r["target_hash"] for r in selected)
        assert all(r["source_hash"] in boards and r["target_hash"] in boards for r in selected)
        assert all("actions" not in r and "reference_actions" not in r for r in selected)
        assert all(len(r["layout_id"]) == 64 and len(r["witness_sha256"]) == 64 for r in selected)
        for row in selected:
            assert row["goal"] and set(row["trials"]) == {"luna", "terra_6_1"}
            luna, terra = row["trials"]["luna"], row["trials"]["terra_6_1"]
            assert len(luna) == len(terra) == 5
            assert [x["action_seed"] for x in luna] == [x["action_seed"] for x in terra]
            assert {x["trial_id"] for x in luna}.isdisjoint(x["trial_id"] for x in terra)
    refs = json.loads((first / "calibration_pairs.json").read_text())["pairs"]
    for context_id in first_manifest["calibration_contexts"]:
        selected = [r for r in refs if r["context_id"] == context_id]
        assert 0 < len(selected) <= 8
        assert len({r["target_hash"] for r in selected}) == len(selected)
        assert all(r["source_hash"] != r["target_hash"] and 1 <= r["k"] <= 16 for r in selected)
        for row in selected:
            for label in ("A", "B"):
                assert set(row["groups"][label]) == {"luna", "terra_6_1"}
                assert all(len(row["groups"][label][model]) == 10 for model in ("luna", "terra_6_1"))
            for model in ("luna", "terra_6_1"):
                assert {x["action_seed"] for x in row["groups"]["A"][model]}.isdisjoint(
                    x["action_seed"] for x in row["groups"]["B"][model])
            for label in ("A", "B"):
                assert {x["trial_id"] for x in row["groups"][label]["luna"]}.isdisjoint(
                    x["trial_id"] for x in row["groups"][label]["terra_6_1"])
    verify_dataset(first)
    verify_dataset(second)
    before = {name: hashlib.sha256((first / name).read_bytes()).hexdigest() for name in names}
    prepare_dataset(root, first, seed=20261103, n_main=2)
    assert before == {name: hashlib.sha256((first / name).read_bytes()).hexdigest() for name in names}
    with pytest.raises(ValueError):
        prepare_dataset(root, first, seed=20261104, n_main=2)
    assert before == {name: hashlib.sha256((first / name).read_bytes()).hexdigest() for name in names}


def test_verify_dataset_rejects_tampered_state_artifact(tmp_path):
    from pathlib import Path

    from carl.icl.reachability_data import prepare_dataset, verify_dataset

    root = Path("data/icl/component_study_100_clean")
    if not (root / "private_witnesses/real_12x12x4/test.json").exists():
        pytest.skip("Real prepared Sokoban witnesses are not available locally")
    output = tmp_path / "corrupted"
    prepare_dataset(root, output, seed=20261103, n_main=2)
    state_file = output / "states.json"
    # Keep valid JSON but alter its bytes; verification must reject corruption.
    state_file.write_text(state_file.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_dataset(output)
