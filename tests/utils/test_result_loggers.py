from __future__ import annotations

import pytest

from carl.planners.base import Experience
from carl.planners.base import SearchInfo
from carl.planners.base import Solution
from carl.utils.result_loggers import SubgoalSearchResultLogger


class FakeSeries:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def append(self, *, step: int, value) -> None:
        self.entries.append({"step": step, "value": value})


class FakeRun(dict):
    def __getitem__(self, key: str) -> FakeSeries:
        if key not in self:
            self[key] = FakeSeries()
        return dict.__getitem__(self, key)


class FakeInnerLogger:
    def __init__(self) -> None:
        self.run = FakeRun()


class FakeCustomLogger:
    def __init__(self, inner_logger: FakeInnerLogger | None = None) -> None:
        self._inner_logger = inner_logger or FakeInnerLogger()

    def return_logger(self) -> FakeInnerLogger:
        return self._inner_logger


class BadCustomLogger:
    def return_logger(self):  # noqa: ANN201
        return object()


def _make_search_info(
    *,
    finished_reason: str,
    low_level_nodes_visited: int,
    subgoals_visited: int,
    tree_size: int,
    tree_depth: int,
    leaf_nodes: int,
    branching_factor: float,
) -> SearchInfo:
    return SearchInfo(
        finished_reason=finished_reason,
        low_level_nodes_visited=low_level_nodes_visited,
        subgoals_visited=subgoals_visited,
        tree_size=tree_size,
        tree_depth=tree_depth,
        leaf_nodes=leaf_nodes,
        branching_factor=branching_factor,
    )


def _make_solved_experience() -> Experience:
    return Experience(
        solution=Solution(
            solved=True,
            subgoal_path=["s1", "s2"],
            action_path=[1, 2, 3],
            subgoal_distance_path=[2, 1],
        ),
        search_info=_make_search_info(
            finished_reason="solved",
            low_level_nodes_visited=4,
            subgoals_visited=2,
            tree_size=7,
            tree_depth=3,
            leaf_nodes=2,
            branching_factor=1.5,
        ),
    )


def _make_unsolved_experience() -> Experience:
    return Experience(
        solution=Solution(solved=False),
        search_info=_make_search_info(
            finished_reason="budget_exceeded",
            low_level_nodes_visited=12,
            subgoals_visited=9,
            tree_size=9,
            tree_depth=4,
            leaf_nodes=3,
            branching_factor=2.0,
        ),
    )


def _last_value(run: FakeRun, key: str):
    return run[key].entries[-1]["value"]


def test_init_raises_when_custom_logger_has_invalid_run():
    with pytest.raises(RuntimeError, match="run"):
        SubgoalSearchResultLogger(
            custom_logger=BadCustomLogger(),  # type: ignore[arg-type]
            budget_logs=[],
            problem_to_solve=1,
        )


def test_node_global_id_uses_env_ids(monkeypatch):
    logger = SubgoalSearchResultLogger(
        custom_logger=FakeCustomLogger(),  # type: ignore[arg-type]
        budget_logs=[],
        problem_to_solve=1,
    )

    monkeypatch.setenv("CARL_HET_GROUP_ID", "7")
    monkeypatch.setenv("CARL_LOCAL_WORKER_ID", "2")
    assert logger.node_global_id == "HG7_W2"

    monkeypatch.delenv("CARL_HET_GROUP_ID")
    monkeypatch.delenv("CARL_LOCAL_WORKER_ID")
    assert logger.node_global_id == ""


def test_log_results_aggregates_metrics_and_finished_reason_rates(monkeypatch):
    monkeypatch.setenv("CARL_HET_GROUP_ID", "1")
    monkeypatch.setenv("CARL_LOCAL_WORKER_ID", "9")
    fake_inner = FakeInnerLogger()
    logger = SubgoalSearchResultLogger(
        custom_logger=FakeCustomLogger(fake_inner),  # type: ignore[arg-type]
        budget_logs=[5, 10],
        problem_to_solve=2,
    )

    logger.log_results([_make_solved_experience()])
    logger.log_results([_make_unsolved_experience()])

    run = fake_inner.run
    solved_key = "solved/HG1_W9"
    finished_solved_key = "finished_reasons/solved/rate/HG1_W9"
    finished_budget_key = "finished_reasons/budget_exceeded/rate/HG1_W9"

    assert _last_value(run, "total_completed_problems") == 2
    assert _last_value(run, "problems/solved/HG1_W9") == 1
    assert _last_value(run, finished_solved_key) == pytest.approx(0.5)
    assert _last_value(run, finished_budget_key) == pytest.approx(0.5)

    solved_stats = _last_value(run, solved_key)
    assert isinstance(solved_stats, dict)
    assert solved_stats["rate/full"] == pytest.approx(0.5)
    assert solved_stats["rate/5_nodes"] == pytest.approx(0.5)
    assert solved_stats["rate/10_nodes"] == pytest.approx(0.5)
    assert solved_stats["rate/5_subgoals"] == pytest.approx(0.5)
    assert solved_stats["rate/10_subgoals"] == pytest.approx(0.5)
    assert solved_stats["average__solved_instances__low_level_solution_len"] == pytest.approx(4.0)
    assert solved_stats["average__all_instances__tree_size"] == pytest.approx(8.0)

    assert len(run["solved_instances__low_level_solution_len"].entries) == 1
    assert len(run["all_instances__tree_size"].entries) == 2
