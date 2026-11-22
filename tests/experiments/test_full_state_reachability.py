"""Exercise the exact first-hit rule and the public reachability fixture."""

import json
from pathlib import Path

import numpy as np
import pytest

from carl.icl.api import RequestFailed
from carl.icl.boards import from_rows, state_hash
from carl.icl.data_sources.sokoban import environment
from carl.icl.evaluation.full_state_reachability import Controller, Tariff, describe, execute_trial
from carl.icl.evaluation.reachability_design import ReachabilityCase, ReachabilityStudy, validated_trajectories, WitnessFile
from carl.icl.types import Board, Probabilities
from scripts.run_full_state_reachability import main

SMOKE = Path("configs/icl/full_state/reachability_smoke.json")
CAMPAIGN = Path("configs/icl/full_state/reachability_campaign.json")
ROWS = ["#####", "#---#", "#@$.#", "#---#", "#####"]


class FixedAction:
    """Exercise simulator decisions, including a failure after a counted attempt."""

    def __init__(self, action: int, *, fail_after: int | None = None) -> None:
        self.action, self.fail_after, self.calls = action, fail_after, 0

    def probabilities(self, state: Board, subgoal: Board) -> Probabilities:
        """Choose one direction, independently of the hidden witness."""
        del state, subgoal
        if self.fail_after == self.calls:
            raise RequestFailed("mock transport failure")
        self.calls += 1
        result = np.zeros(4)
        result[self.action] = 1.0
        return result


def _case(target: list[str], k: int) -> ReachabilityCase:
    return ReachabilityCase(trajectory="toy", layout="toy", index=0, k=k, seed=7,
                            source=ROWS, target=target, source_sha256=state_hash(from_rows(ROWS)),
                            target_sha256=state_hash(from_rows(target)))


def test_first_exact_hit_and_failed_partial_attempt() -> None:
    env = environment(from_rows(ROWS))
    target = ["#####", "#@--#", "#-$.#", "#---#", "#####"]
    case = _case(target, 3)
    hit = execute_trial(case, Controller(FixedAction(0), {"model": "up"}), env)
    assert (hit.first_hit, hit.actions, hit.reason) == (1, [0], "exact_target")
    terminal = execute_trial(case, Controller(FixedAction(3), {"model": "right"}), env)
    assert (terminal.first_hit, terminal.actions, terminal.reason) == (None, [3], "terminal_before_target")
    failed = execute_trial(case, Controller(FixedAction(2, fail_after=1), {"model": "left"}), env)
    assert failed.status == "error"
    assert failed.first_hit is None and failed.accounting["attempted_simulator_actions"] == 1


def test_full_campaign_preflight_and_live_gate(tmp_path: Path) -> None:
    study = ReachabilityStudy.model_validate_json(CAMPAIGN.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in describe(study, {"openai-four-shot": Tariff(input_usd=1, output_usd=5)})["models"]}
    assert study.trials_per_point == 100 and len(rows) == 5
    assert rows["ministral-four-shot"]["offsets"] == [1, 2, 4, 8, 16]
    assert rows["ministral-four-shot"]["route"]["api"] == "chat_completions"
    assert rows["openai-four-shot"]["max_policy_decisions"] == 7560
    assert rows["openai-four-shot"]["cost_estimate_usd"] == pytest.approx(125.7984)
    for flags in ([], ["--execute-live"], ["--execute-live", "--model", "bert"]):
        with pytest.raises(SystemExit) as error:
            main(["--config", str(CAMPAIGN), "--output", str(tmp_path / "no-dispatch"), *flags])
        assert error.value.code == 1
    assert not any(tmp_path.iterdir())


def test_public_fixture_wilson_resume_and_schema_validation(tmp_path: Path) -> None:
    output = tmp_path / "results"
    args = ["--config", str(SMOKE), "--output", str(output)]
    assert main(args) == 0
    paths = sorted((output / "trials").rglob("*.json"))
    first_run = {path: path.read_bytes() for path in paths}
    points = json.loads((output / "report.json").read_text(encoding="utf-8"))["points"]
    assert len(paths) == 20 and len(points) == 5
    assert [point["successes"] for point in points] == [2, 3, 2, 3, 1]
    first = points[0]
    assert first["wilson_low_pct"] == pytest.approx(15.0038989152)
    assert first["wilson_high_pct"] == pytest.approx(84.9961010848)
    assert all(p["status"] == "complete" and p["valid"] == 4 and p["physical_requests"] == 0 for p in points)
    assert (output / "reachability.svg").is_file() and (output / "points.csv").is_file()
    train, test = (WitnessFile.model_validate_json((output / f"{split}-fixture.json").read_text(encoding="utf-8"))
                   for split in ("train", "test"))
    assert len(validated_trajectories(train, "train")) == len(validated_trajectories(test, "test")) == 4
    with pytest.raises(ValueError, match="expected a 12x12 board with 10 boxes"):
        validated_trajectories(train.model_copy(update={"split": "test"}), "test")
    assert main(["--report-only", "--output", str(output)]) == 0
    assert main(args) == 0
    assert {path: path.read_bytes() for path in paths} == first_run
    override = tmp_path / "override.json"
    override.write_text(SMOKE.read_text(encoding="utf-8").replace("20261122", "101"), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["--config", str(override), "--output", str(output)])
    assert error.value.code == 1
    assert {path: path.read_bytes() for path in paths} == first_run
