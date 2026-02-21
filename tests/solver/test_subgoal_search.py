from __future__ import annotations

from dataclasses import dataclass

import pytest

from carl.planners.base import Experience
from carl.planners.base import FinishReason
from carl.planners.base import Planner
from carl.planners.base import SearchInfo
from carl.planners.base import Solution
from carl.solver.nodes import SearchTreeNode
from carl.solver.nodes import ValidationResult
from carl.solver.subgoal_search import Solver


def _make_planner_class(initial_k: int | None, pre_seen: set[str] | None = None):
    class DummyPlanner(Planner):
        last_instance: "DummyPlanner | None" = None

        def __init__(self, root_state):
            self.root_state = root_state
            self.root_node = SearchTreeNode(
                state=root_state,
                value=0.0,
                low_level_path=[],
                parent_node=None,
                next_expand_with_k_generator=initial_k,
                metadata={"depth": 0},
            )
            self.queue = [self.root_node] if initial_k is not None else []
            self.added: list[SearchTreeNode] = []
            self._seen = {root_state}
            self._pre_seen = set(pre_seen or set())
            type(self).last_instance = self

        def add(self, node: SearchTreeNode) -> None:
            node.next_expand_with_k_generator = node.metadata.get("next_k")
            self._seen.add(node.state)
            self.added.append(node)
            self.queue.append(node)

        def get(self) -> SearchTreeNode | None:
            if not self.queue:
                return None
            return self.queue.pop(0)

        def is_seen(self, state) -> bool:
            return state in self._seen or state in self._pre_seen

        def get_solution_data(self, solving_node: SearchTreeNode | None, search_info: SearchInfo) -> Experience:
            search_info.solving_node = solving_node
            if solving_node is None:
                solution = Solution(solved=False)
            else:
                solution = Solution(
                    solved=True,
                    subgoal_path=[],
                    action_path=[],
                    subgoal_distance_path=[],
                )
            return Experience(solution=solution, search_info=search_info)

    return DummyPlanner


@dataclass
class DummyGenerator:
    generator_k_list: list[int]
    mapping: dict[str, list[tuple[str, dict[str, int]]]]

    def construct_network(self) -> None:
        pass

    def get_subgoals(self, node: SearchTreeNode):
        return self.mapping.get(node.state, [])


class RecordingValidator:
    def __init__(self, responses: dict[tuple[str, str], ValidationResult]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, int | None]] = []

    def construct_network(self) -> None:
        pass

    def is_valid(self, state, subgoal, **kwargs) -> ValidationResult:
        self.calls.append((state, subgoal, kwargs.get("steps_limit")))
        return self.responses[(state, subgoal)]


class RecordingValue:
    def __init__(self, values: dict[str, float]) -> None:
        self.values = values
        self.calls: list[str] = []

    def construct_network(self) -> None:
        pass

    def get_value(self, state: str) -> float:
        self.calls.append(state)
        return self.values[state]


def test_solve_finishes_with_nothing_to_expand(monkeypatch):
    monkeypatch.setattr("carl.solver.subgoal_search.ensure_high_recursion_limit", lambda: None)
    planner_class = _make_planner_class(initial_k=None)
    generator = DummyGenerator(generator_k_list=[2, 3], mapping={})
    validator = RecordingValidator({})
    value = RecordingValue({})

    solver = Solver(
        max_nodes=8,
        planner_class=planner_class,
        subgoal_generator=generator,  # type: ignore[arg-type]
        validator=validator,  # type: ignore[arg-type]
        value_function=value,  # type: ignore[arg-type]
    )

    experience = solver.solve("root")

    assert experience.solution.solved is False
    assert experience.search_info.finished_reason == FinishReason.NOTHING_TO_EXPAND.value
    assert experience.search_info.low_level_nodes_visited == 0
    assert experience.search_info.high_level_nodes_valid == 0
    assert experience.search_info.high_level_nodes_unreachable == 0
    assert experience.search_info.subgoals_reachable_count_per_k == {2: 0, 3: 0}
    assert experience.search_info.subgoals_unreachable_count_per_k == {2: 0, 3: 0}
    assert experience.search_info.subgoals_reachable_rate_per_k == {2: 0, 3: 0}
    assert validator.calls == []
    assert value.calls == []


def test_solve_tracks_per_k_metrics_and_steps_limit(monkeypatch):
    monkeypatch.setattr("carl.solver.subgoal_search.ensure_high_recursion_limit", lambda: None)
    planner_class = _make_planner_class(initial_k=2, pre_seen={"seen_subgoal"})
    generator = DummyGenerator(
        generator_k_list=[2, 3],
        mapping={
            "root": [
                ("a", {"next_k": 3}),
                ("bad", {"next_k": 3}),
                ("seen_subgoal", {"next_k": 3}),
            ],
            "a": [("goal", {"next_k": 2}), ("ignored_after_solution", {"next_k": 2})],
        },
    )
    validator = RecordingValidator(
        {
            ("root", "a"): ValidationResult(True, False, ["root->a"], 2, "a"),
            ("root", "bad"): ValidationResult(False, False, ["root->bad"], 1, "bad"),
            ("a", "goal"): ValidationResult(True, True, ["a->goal"], 3, "goal"),
        }
    )
    value = RecordingValue({"a": 0.3, "goal": 0.0})

    solver = Solver(
        max_nodes=20,
        planner_class=planner_class,
        subgoal_generator=generator,  # type: ignore[arg-type]
        validator=validator,  # type: ignore[arg-type]
        value_function=value,  # type: ignore[arg-type]
    )

    experience = solver.solve("root")
    planner = planner_class.last_instance
    assert planner is not None

    assert experience.solution.solved is True
    assert experience.search_info.finished_reason == FinishReason.SOLVED.value
    assert experience.search_info.low_level_nodes_visited == 6
    assert experience.search_info.high_level_nodes_valid == 2
    assert experience.search_info.high_level_nodes_unreachable == 1

    assert experience.search_info.subgoals_reachable_count_per_k == {2: 1, 3: 1}
    assert experience.search_info.subgoals_unreachable_count_per_k == {2: 1, 3: 0}
    assert experience.search_info.subgoals_reachable_rate_per_k == {2: 0.5, 3: 1.0}

    assert validator.calls == [
        ("root", "a", 2),
        ("root", "bad", 2),
        ("a", "goal", 3),
    ]
    assert value.calls == ["a", "goal"]
    assert [node.state for node in planner.added] == ["a", "goal"]


def test_solve_budget_exceeded_when_nodes_limit_crossed(monkeypatch):
    monkeypatch.setattr("carl.solver.subgoal_search.ensure_high_recursion_limit", lambda: None)
    planner_class = _make_planner_class(initial_k=2)
    generator = DummyGenerator(
        generator_k_list=[2],
        mapping={"root": [("far", {"next_k": 2})]},
    )
    validator = RecordingValidator(
        {("root", "far"): ValidationResult(False, False, ["root->far"], 5, "far")}
    )
    value = RecordingValue({})

    solver = Solver(
        max_nodes=3,
        planner_class=planner_class,
        subgoal_generator=generator,  # type: ignore[arg-type]
        validator=validator,  # type: ignore[arg-type]
        value_function=value,  # type: ignore[arg-type]
    )

    experience = solver.solve("root")

    assert experience.solution.solved is False
    assert experience.search_info.finished_reason == FinishReason.BUDGET_EXCEEDED.value
    assert experience.search_info.low_level_nodes_visited == 5
    assert experience.search_info.high_level_nodes_valid == 0
    assert experience.search_info.high_level_nodes_unreachable == 1
    assert experience.search_info.subgoals_reachable_count_per_k == {2: 0}
    assert experience.search_info.subgoals_unreachable_count_per_k == {2: 1}
    assert experience.search_info.subgoals_reachable_rate_per_k == {2: 0.0}
