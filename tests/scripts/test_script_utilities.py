import csv
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import pytest

from carl.planners.base import Experience
from carl.planners.base import SearchInfo
from carl.planners.base import Solution
from carl.solver.nodes import SearchTreeNode


def _load_script_module(script_name: str):
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    script_path = scripts_dir / f"{script_name}.py"
    module_name = f"test_script_{script_name}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_solved_experience(state_value: int) -> Experience:
    root = SearchTreeNode(state_value, 0.0, [], None, metadata={"depth": 0})
    return Experience(
        solution=Solution(
            solved=True,
            subgoal_path=[],
            action_path=[],
            subgoal_distance_path=[],
        ),
        search_info=SearchInfo(finished_reason="solved", solving_node=root),
    )


def _make_unsolved_experience() -> Experience:
    return Experience(
        solution=Solution(solved=False),
        search_info=SearchInfo(finished_reason="budget_exceeded", solving_node=None),
    )


def _run_main(module, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def test_gather_handles_mixed_valid_invalid_and_corrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gather = _load_script_module("gather_k841_solutions")

    good = tmp_path / "good.joblib"
    empty = tmp_path / "empty.joblib"
    bad = tmp_path / "bad.joblib"

    payload = [
        _make_solved_experience(1),
        _make_unsolved_experience(),
        "junk_record",
        123,
    ]
    joblib.dump(payload, good)
    joblib.dump([], empty)
    bad.write_text("this is not a joblib payload", encoding="utf-8")

    output_dir = tmp_path / "out"
    _run_main(
        gather,
        [
            "gather_k841_solutions.py",
            "--input-glob",
            str(tmp_path / "*.joblib"),
            "--output-dir",
            str(output_dir),
        ],
        monkeypatch,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    merged_all = joblib.load(output_dir / "merged_all.joblib")
    merged_solved = joblib.load(output_dir / "merged_solved.joblib")

    assert summary["matched_files_count"] == 3
    assert summary["loaded_files_count"] == 2
    assert summary["failed_files_count"] == 1
    assert summary["counts"]["all_records"] == 2
    assert summary["counts"]["solved"] == 1
    assert summary["counts"]["unsolved"] == 1
    assert summary["counts"]["unknown_items_ignored"] == 2
    assert len(merged_all) == 2
    assert len(merged_solved) == 1
    assert len(summary["failures"]) == 1
    assert summary["failures"][0]["path"].endswith("bad.joblib")


def test_gather_handles_empty_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gather = _load_script_module("gather_k841_solutions")

    empty = tmp_path / "only_empty.joblib"
    joblib.dump([], empty)

    output_dir = tmp_path / "out_empty"
    _run_main(
        gather,
        [
            "gather_k841_solutions.py",
            "--input-glob",
            str(empty),
            "--output-dir",
            str(output_dir),
        ],
        monkeypatch,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["all_records"] == 0
    assert summary["counts"]["solved"] == 0
    assert summary["counts"]["unsolved"] == 0
    assert summary["counts"]["unknown_items_ignored"] == 0
    assert joblib.load(output_dir / "merged_all.joblib") == []
    assert joblib.load(output_dir / "merged_solved.joblib") == []


def test_inspect_sampling_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    inspect_mod = _load_script_module("inspect_k841_trajectories")

    dataset = tmp_path / "sample.joblib"
    experiences = [_make_solved_experience(i) for i in range(30)]
    joblib.dump(experiences, dataset)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out3 = tmp_path / "run3"

    base_args = [
        "inspect_k841_trajectories.py",
        "--input-glob",
        str(dataset),
        "--n",
        "8",
    ]

    _run_main(inspect_mod, base_args + ["--seed", "7", "--output-dir", str(out1)], monkeypatch)
    _run_main(inspect_mod, base_args + ["--seed", "7", "--output-dir", str(out2)], monkeypatch)
    _run_main(inspect_mod, base_args + ["--seed", "99", "--output-dir", str(out3)], monkeypatch)

    rows1 = list(csv.DictReader((out1 / "manual_inspection.csv").open(encoding="utf-8")))
    rows2 = list(csv.DictReader((out2 / "manual_inspection.csv").open(encoding="utf-8")))
    rows3 = list(csv.DictReader((out3 / "manual_inspection.csv").open(encoding="utf-8")))

    sample1 = [(row["sample_rank"], row["source_index"]) for row in rows1]
    sample2 = [(row["sample_rank"], row["source_index"]) for row in rows2]
    sample3 = [(row["sample_rank"], row["source_index"]) for row in rows3]

    assert sample1 == sample2
    assert sample1 != sample3


def test_grid_parameter_expansion_and_parsing():
    grid_mod = _load_script_module("build_generator_targets_grid")

    configs = grid_mod._build_grid(
        modes=("sliding_window", "k_offsets", "hierarchical_raw"),
        k_values=(1, 2),
        offsets_sets=((-1, 0, 1),),
        radii=(0, 1),
        distance_intervals=(None, (8,)),
    )

    assert len(configs) == 8
    mode_counts: dict[str, int] = {}
    for config in configs:
        mode_counts[config.mode] = mode_counts.get(config.mode, 0) + 1

    assert mode_counts == {
        "sliding_window": 2,
        "k_offsets": 4,
        "hierarchical_raw": 2,
    }

    assert grid_mod._parse_int_csv("1,2,3") == (1, 2, 3)
    assert grid_mod._parse_optional_int_csv("none") is None
    assert grid_mod._parse_optional_int_csv("4,8") == (4, 8)


def test_estimate_capacity_rejects_corrupted_artifact(tmp_path: Path):
    estimate = _load_script_module("estimate_batch_capacity")

    bad_artifact = tmp_path / "bad.json"
    bad_artifact.write_text(json.dumps({"examples_count": 5}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        estimate._load_artifact(bad_artifact)

    with pytest.raises(ValueError, match="positive"):
        estimate._parse_int_list("16,0,32")


def test_compare_trajectory_characteristics_outputs_expected_budget_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    compare_mod = _load_script_module("compare_trajectory_characteristics")

    root_a = SearchTreeNode("root_a", 0.0, [], None, metadata={"depth": 0})
    root_b = SearchTreeNode("root_b", 0.0, [], None, metadata={"depth": 0})
    root_c = SearchTreeNode("root_c", 0.0, [], None, metadata={"depth": 0})
    root_d = SearchTreeNode("root_d", 0.0, [], None, metadata={"depth": 0})

    imitation_payload = [
        Experience(
            solution=Solution(
                solved=True,
                subgoal_path=["g1", "g2", "g3"],
                action_path=[0, 1, 2],
                subgoal_distance_path=[8, 4, 1],
            ),
            search_info=SearchInfo(
                finished_reason="solved",
                low_level_nodes_visited=40,
                tree_size=40,
                tree_depth=5,
                leaf_nodes=12,
                branching_factor=2.0,
                subgoals_visited=3,
                solving_node=root_a,
            ),
        ),
        Experience(
            solution=Solution(solved=False),
            search_info=SearchInfo(
                finished_reason="budget_exceeded",
                low_level_nodes_visited=180,
                tree_size=180,
                tree_depth=8,
                leaf_nodes=30,
                branching_factor=1.6,
                subgoals_visited=7,
                solving_node=root_b,
            ),
        ),
    ]

    scratch_payload = [
        Experience(
            solution=Solution(
                solved=True,
                subgoal_path=["g1", "g2"],
                action_path=[0, 1, 2, 3, 4],
                subgoal_distance_path=[8, 8],
            ),
            search_info=SearchInfo(
                finished_reason="solved",
                low_level_nodes_visited=120,
                tree_size=120,
                tree_depth=7,
                leaf_nodes=25,
                branching_factor=1.4,
                subgoals_visited=4,
                solving_node=root_c,
            ),
        ),
        Experience(
            solution=Solution(solved=False),
            search_info=SearchInfo(
                finished_reason="budget_exceeded",
                low_level_nodes_visited=260,
                tree_size=260,
                tree_depth=10,
                leaf_nodes=45,
                branching_factor=1.2,
                subgoals_visited=12,
                solving_node=root_d,
            ),
        ),
    ]

    imitation_file = tmp_path / "imitation.joblib"
    scratch_file = tmp_path / "scratch.joblib"
    joblib.dump(imitation_payload, imitation_file)
    joblib.dump(scratch_payload, scratch_file)

    output_dir = tmp_path / "comparison"
    _run_main(
        compare_mod,
        [
            "compare_trajectory_characteristics.py",
            "--imitation-glob",
            str(imitation_file),
            "--scratch-glob",
            str(scratch_file),
            "--node-budget",
            "64",
            "--node-budget",
            "128",
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ],
        monkeypatch,
    )

    csv_path = output_dir / "trajectory_stats.csv"
    md_path = output_dir / "trajectory_stats.md"
    assert csv_path.exists()
    assert md_path.exists()

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    budget_rows = [
        row for row in rows if row["type"] == "budget_rate" and row["metric"] == "solved_within_low_level_nodes"
    ]
    assert budget_rows

    imitation_64 = next(
        row for row in budget_rows if row["group"] == "imitation" and row["budget"] == "64"
    )
    scratch_64 = next(
        row for row in budget_rows if row["group"] == "scratch" and row["budget"] == "64"
    )
    scratch_128 = next(
        row for row in budget_rows if row["group"] == "scratch" and row["budget"] == "128"
    )

    assert imitation_64["count"] == "1"
    assert imitation_64["total"] == "2"
    assert float(imitation_64["rate"]) == pytest.approx(0.5)

    assert scratch_64["count"] == "0"
    assert scratch_64["total"] == "2"
    assert float(scratch_64["rate"]) == pytest.approx(0.0)

    assert scratch_128["count"] == "1"
    assert scratch_128["total"] == "2"
    assert float(scratch_128["rate"]) == pytest.approx(0.5)
