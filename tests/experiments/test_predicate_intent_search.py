"""Compare the predicate planner modes on deterministic, replayable Sokoban cases."""

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

import pytest

from carl.icl.api import BudgetExceeded
from carl.icl.boards import to_rows
from carl.icl.config import ExperimentConfig, Limits
from carl.icl.data_sources.sokoban import environment
from carl.icl.experiment import create_inference_api
from carl.icl.intent_annotations import prepare_annotations
from carl.icl.intent_domain import TASK_GOAL, Intent, SokobanObservation, SokobanSnapshot, decode_intent, replay_fragment
from carl.icl.intents.generation.data import read_records
from carl.icl.llm_components.predicate import create_predicate_components
from carl.icl.luna_intent_rollout import Action, ActionAttempt
from carl.icl.planning.intent_execution import SokobanIntentExecutor
from carl.icl.planning.intent_search import IntentSearchRuntime, IntentSearchSettings, search_intents
from carl.icl.planning.intent_value import fit_value, load_value
from carl.icl.types import Board, JSONObject
from scripts import run_predicate_icl


START = SokobanSnapshot(("#######", "#@$-.-#", "#-----#", "#-----#", "#-----#", "#-----#", "#######"))
HISTORY = (ActionAttempt(Action.LEFT, True),)
PlannerMode = Literal["direct", "one-step", "full-search"]
Decision = Callable[[SokobanObservation, Intent | None, Sequence[ActionAttempt]], Action]
Proposal = Callable[[SokobanObservation, str, float | None], Sequence[Intent]]
Value = Callable[[SokobanObservation, str], float]


def _intent(predicate: str, row: int, col: int, *, preserve: list[JSONObject] | None = None) -> Intent:
    return decode_intent({"effects": [{"predicate": predicate, "row": row, "col": col}],
                          "preserve": preserve or [], "instruction": None})


def _player(observation: SokobanObservation) -> tuple[int, int]:
    return next((row, col) for row, tiles in enumerate(observation.rows) for col, tile in enumerate(tiles)
                if tile in "@+")


def _json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class ObservedDecision:
    """The actual observation and controller memory before emitting a primitive action."""

    instance: int
    index: int
    observation: SokobanObservation
    intent: Intent | None
    memory: tuple[ActionAttempt, ...]


class StatefulFixturePolicy:
    """Each instance has a private mutable call counter for detecting branch leaks."""

    def __init__(self, serial: int, decisions: list[ObservedDecision], decide: Decision) -> None:
        self.serial = serial
        self.decisions = decisions
        self.decide = decide
        self.index = 0

    def render_action(self, observation: SokobanObservation, intent: Intent, memory: Sequence[ActionAttempt]) -> str:
        return "fixture intent decision"

    def render_task_action(self, observation: SokobanObservation, goal: str, memory: Sequence[ActionAttempt]) -> str:
        assert goal == TASK_GOAL
        return "fixture task decision"

    def _choose(self, observation: SokobanObservation, intent: Intent | None, memory: Sequence[ActionAttempt]) -> str:
        self.index += 1
        self.decisions.append(ObservedDecision(self.serial, self.index, observation, intent, tuple(memory)))
        return json.dumps({"action": self.decide(observation, intent, memory).value})

    def choose_action(self, observation: SokobanObservation, intent: Intent, memory: Sequence[ActionAttempt]) -> str:
        return self._choose(observation, intent, memory)

    def choose_task_action(self, observation: SokobanObservation, goal: str, memory: Sequence[ActionAttempt]) -> str:
        assert goal == TASK_GOAL
        return self._choose(observation, None, memory)


def _value(_observation: SokobanObservation, goal: str) -> float:
    assert goal == TASK_GOAL
    return 0.5


def _runtime(
    decide: Decision, propose: Proposal, *, value: Value = _value,
    observe: Callable[[Board], SokobanObservation] | None = None,
) -> tuple[IntentSearchRuntime, list[ObservedDecision]]:
    decisions: list[ObservedDecision] = []
    counters = {"logical_requests": 0, "physical_requests": 0, "policy_calls": 0, "generator_calls": 0, "value_calls": 0}
    serial = 0

    def count(role: str) -> None:
        counters["logical_requests"] += 1
        counters[f"{role}_calls"] += 1

    def action(observation: SokobanObservation, intent: Intent | None, memory: Sequence[ActionAttempt]) -> Action:
        count("policy")
        return decide(observation, intent, memory)

    def factory(_seed: int) -> StatefulFixturePolicy:
        nonlocal serial
        serial += 1
        return StatefulFixturePolicy(serial, decisions, action)

    def proposal(observation: SokobanObservation, goal: str, scale: float | None) -> Sequence[Intent]:
        count("generator")
        assert goal == TASK_GOAL
        return propose(observation, goal, scale)

    def estimate(observation: SokobanObservation, goal: str) -> float:
        count("value")
        return value(observation, goal)

    executor = SokobanIntentExecutor(factory, observe=observe)
    return IntentSearchRuntime(executor=executor, propose=proposal, value=estimate, metrics=lambda: counters), decisions


def _down(_observation: SokobanObservation, _intent: Intent | None, _memory: Sequence[ActionAttempt]) -> Action:
    return Action.DOWN


def _right(_observation: SokobanObservation, _intent: Intent | None, _memory: Sequence[ActionAttempt]) -> Action:
    return Action.RIGHT


def _settings(
    *, scales: tuple[float | None, ...] = (0.8,), budget: int = 12, cap: int = 3, max_proposals: int = 1,
    discount: float = 0.99,
) -> IntentSearchSettings:
    return IntentSearchSettings(scales=scales, action_budget=budget, rollout_cap=cap, max_expansions=8,
                                wall_seconds=60, max_proposals=max_proposals, trials_per_proposal=1,
                                discount=discount, seed=73)


@pytest.fixture(name="cli_files")
def fixture_cli_files(tmp_path: Path) -> tuple[list[str], Path, Path, Path]:
    trajectories = []
    queries: dict[str, Path] = {}
    for split, walls in (("train", "#-#---#"), ("validation", "#-##--#"), ("test", "#-----#")):
        rows = [*START.rows[:4], walls, *START.rows[5:]]
        group = f"group-{split}"
        trajectories.append({"source_id": split, "task_group": group, "split": split,
                             "start_observation": rows, "actions": [3, 3]})
        queries[split] = _json(tmp_path / f"{split}.json", {
            "observation": {"rows": rows}, "task_goal": TASK_GOAL, "scale": None, "split": split, "task_group": group,
        })
    source = _json(tmp_path / "demonstrations.json", {"trajectories": trajectories})
    annotation = tmp_path / "annotations"
    assert prepare_annotations(source, annotation, max_offset=1, seed=17)["accepted"] == 6
    data = annotation / "D_A.jsonl"
    anchor = next(row for row in read_records(data, "validation")
                  if row["start_observation"] == trajectories[1]["start_observation"])
    config = _json(tmp_path / "config.json", {
        "model": "sokoban-planning-smoke", "seed": 73, "action_budget": 12, "max_expansions": 8,
        "cache_dir": str(tmp_path / "cache"), "policy_tokens": 64, "generator_tokens": 256,
        "predicate_icl": {"validation_id": anchor["record_id"], "policy_shots": 0, "generator_shots": 0,
                          "context_limit": 8192, "scale_arm": "unscaled", "planning": {"rollout_cap": 2}},
    })
    mock = _json(tmp_path / "mock.json", {"policy": '{"action":"RIGHT"}',
                                          "generator": json.dumps(_intent("box_at", 1, 4)), "value": '{"value":0.5}'})
    arguments = ["--config", str(config), "--data", str(data), "--manifest", str(tmp_path / "support.json")]
    return arguments, queries["validation"], queries["test"], mock


def test_public_cli_three_modes_share_controller_and_replay_the_same_solution(
    tmp_path: Path, cli_files: tuple[list[str], Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, validation, held_out, mock = cli_files

    def invoke(command: str, *extra: str) -> None:
        monkeypatch.setattr(sys, "argv", ["run_predicate_icl", command, *arguments, *extra])
        assert run_predicate_icl.main() == 0

    invoke("freeze")
    reports = {}
    for mode in ("direct", "one-step", "full-search"):
        receipt = tmp_path / f"{mode}-validation.json"
        result = tmp_path / f"{mode}-test.json"
        options = ("--planning-mode", mode, "--mode", "mock", "--mock-response", str(mock))
        invoke("search", *options, "--query", str(validation), "--output", str(receipt))
        invoke("search", *options, "--query", str(held_out), "--validation-result", str(receipt), "--output", str(result))
        report = json.loads(result.read_text(encoding="utf-8"))
        reports[mode] = report
        assert report["solved"] and report["action_path"] == [3, 3]
        expected = replay_fragment(START, report["action_path"])[-1]
        assert environment(expected.restore()).is_solved(expected.restore())
        counts = report["accounting"]
        assert counts["physical_requests"] == 0
        assert counts["executed_plan_actions"] == 2
        assert counts["attempted_simulator_actions"] <= 12
        assert counts["logical_model_calls"] == sum(counts[f"logical_{role}_calls"] for role in ("policy", "generator", "value"))
    assert len({entry["components"]["policy"]["fingerprint"] for entry in reports.values()}) == 1
    direct, one_step, full = (reports[key] for key in ("direct", "one-step", "full-search"))
    assert direct["accounting"]["logical_generator_calls"] == direct["accounting"]["logical_value_calls"] == 0
    assert direct["queue_pops"] == one_step["queue_pops"] == []
    assert one_step["accounting"]["attempted_simulator_actions"] == 4
    assert full["accounting"]["attempted_simulator_actions"] == 2 and full["queue_pops"]


@pytest.mark.parametrize(("values", "first"), [((0.25, 0.75), 2), ((0.75, 0.75), 1)])
def test_queue_prioritizes_scale_then_value_then_insertion(values: tuple[float, float], first: int) -> None:
    def propose(observation: SokobanObservation, _goal: str, scale: float | None) -> Sequence[Intent]:
        if _player(observation) == (1, 1) and scale == 0.8:
            return [_intent("player_at", 2, 1), _intent("player_at", 3, 1)]
        return []

    def value(observation: SokobanObservation, _goal: str) -> float:
        return {1: 0.5, 2: values[0], 3: values[1]}[_player(observation)[0]]

    runtime, decisions = _runtime(_down, propose, value=value)
    result = search_intents(START, TASK_GOAL, mode="full-search", runtime=runtime,
                            settings=_settings(scales=(0.2, 0.8), max_proposals=2), initial_memory=HISTORY)
    pops = [(pop.node, pop.scale, pop.value) for pop in result.queue_pops]
    assert pops[:3] == [(0, 0.8, 0.5), (first, 0.8, values[first - 1]), (3 - first, 0.8, values[2 - first])]
    assert {node for node, scale, _ in pops[3:] if scale == 0.2} == {0, 1, 2}
    assert result.finish_reason == "frontier_exhausted"
    assert result.accounting["attempted_simulator_actions"] == 3
    assert [entry.index for entry in decisions] == [1, 1, 2]
    assert decisions[0].instance != decisions[1].instance == decisions[2].instance
    assert decisions[0].observation == decisions[1].observation == START.observation
    assert decisions[0].memory == decisions[1].memory == HISTORY
    assert decisions[2].memory == (*HISTORY, ActionAttempt(Action.DOWN, False))


def test_one_step_applies_discount_while_probes_share_the_parent() -> None:
    def propose(observation: SokobanObservation, _goal: str, _scale: float | None) -> Sequence[Intent]:
        return [_intent("player_at", 2, 1), _intent("player_at", 3, 1)] if _player(observation) == (1, 1) else []

    def value(observation: SokobanObservation, _goal: str) -> float:
        return {1: 0.0, 2: 0.55, 3: 0.90}[_player(observation)[0]]

    runtime, decisions = _runtime(_down, propose, value=value)
    result = search_intents(START, TASK_GOAL, mode="one-step", runtime=runtime,
                            settings=_settings(max_proposals=2, discount=0.4), initial_memory=HISTORY)
    assert result.action_path == [1]
    assert not result.queue_pops
    assert [attempt.phase for attempt in result.attempts] == ["probe", "probe", "commit"]
    assert result.accounting["attempted_simulator_actions"] == 4
    assert result.accounting["executed_plan_actions"] == 1
    assert all(entry.memory == HISTORY and entry.observation == START.observation
               for entry in decisions if entry.index == 1)


def test_one_step_replans_after_a_safe_committed_prefix_exhausts_its_rollout_cap() -> None:
    candidate = _intent("box_at", 1, 4)
    runtime, decisions = _runtime(_right, lambda _obs, _goal, _scale: [candidate])
    result = search_intents(START, TASK_GOAL, mode="one-step", runtime=runtime, settings=_settings(cap=1))
    assert result.solved and result.intent_reached and result.action_path == [3, 3]
    assert [attempt.phase for attempt in result.attempts] == ["probe", "commit", "probe", "commit"]
    assert result.attempts[1].disposition == "committed_incomplete"
    assert decisions[2].memory == decisions[3].memory == (ActionAttempt(Action.RIGHT, False),)
    assert result.accounting["attempted_simulator_actions"] == 4
    assert result.accounting["executed_plan_actions"] == 2


def test_rejected_branch_charges_no_op_and_cut_branch_keeps_parent_memory() -> None:
    rejected = _intent("player_at", 2, 1, preserve=[{"predicate": "box_at", "row": 1, "col": 2}])
    candidate = _intent("box_at", 1, 4)

    def decide(_obs: SokobanObservation, intent: Intent | None, memory: Sequence[ActionAttempt]) -> Action:
        return Action.UP if intent == rejected and tuple(memory) == HISTORY else Action.RIGHT

    runtime, decisions = _runtime(decide, lambda _obs, _goal, _scale: [rejected, candidate])
    result = search_intents(START, TASK_GOAL, mode="full-search", runtime=runtime,
                            settings=_settings(budget=3, cap=4, max_proposals=2), initial_memory=HISTORY)
    assert not result.solved and result.finish_reason == "action_budget"
    assert [(attempt.parent, attempt.node, attempt.status, list(attempt.actions)) for attempt in result.attempts] == [
        (0, None, "preservation_violated", [0, 3]), (0, None, "action_limit", [3]),
    ]
    assert [entry.index for entry in decisions] == [1, 2, 1]
    assert decisions[0].observation == decisions[1].observation == decisions[2].observation == START.observation
    assert decisions[1].memory == (*HISTORY, ActionAttempt(Action.UP, True))
    assert decisions[2].memory == HISTORY
    assert result.accounting["attempted_simulator_actions"] == 3
    assert result.accounting["attempted_no_op_actions"] == 1
    assert result.accounting["logical_policy_calls"] == 3
    assert result.accounting["executed_plan_actions"] == 0


def test_duplicate_snapshot_is_retained_only_if_controller_memory_differs() -> None:
    initial = SokobanSnapshot((*START.rows[:3], "#--*--#", *START.rows[4:]))
    short = _intent("player_at", 2, 1)
    same_history = _intent("player_at", 2, 1, preserve=[{"predicate": "box_at", "row": 1, "col": 2}])
    different_history = _intent("player_at", 2, 1, preserve=[{"predicate": "boxes_on_goals_at_least", "count": 1}])

    def propose(observation: SokobanObservation, _goal: str, _scale: float | None) -> Sequence[Intent]:
        return [short, same_history, different_history] if observation == initial.observation else []

    def decide(_obs: SokobanObservation, intent: Intent | None, memory: Sequence[ActionAttempt]) -> Action:
        return Action.UP if intent == different_history and not memory else Action.DOWN

    runtime, _ = _runtime(decide, propose)
    result = search_intents(initial, TASK_GOAL, mode="full-search", runtime=runtime,
                            settings=_settings(max_proposals=3))
    first, duplicate, delayed = result.attempts
    assert first.endpoint == duplicate.endpoint == delayed.endpoint
    assert [attempt.disposition for attempt in result.attempts] == ["accepted", "duplicate", "accepted"]
    assert first.node is not None and duplicate.node is None and delayed.node is not None
    assert result.nodes[first.node].state.memory == (ActionAttempt(Action.DOWN, False),)
    assert result.nodes[delayed.node].state.memory == (ActionAttempt(Action.UP, True), ActionAttempt(Action.DOWN, False))
    assert result.accounting["attempted_simulator_actions"] == 4


def test_api_budget_returns_the_last_accepted_replayable_prefix() -> None:
    def propose(observation: SokobanObservation, _goal: str, _scale: float | None) -> Sequence[Intent]:
        if observation == START.observation:
            return [_intent("box_at", 1, 3)]
        raise BudgetExceeded("fixture logical request cap")

    runtime, _ = _runtime(_right, propose)
    result = search_intents(START, TASK_GOAL, mode="full-search", runtime=runtime, settings=_settings())
    assert result.finish_reason == "api_budget" and not result.solved
    assert result.action_path == [3] and result.intent_reached
    assert result.endpoint == replay_fragment(START, [3])[-1]
    assert result.accounting["attempted_simulator_actions"] == result.accounting["executed_plan_actions"] == 1


def test_actual_mock_cache_and_generator_retry_use_logical_budget(tmp_path: Path) -> None:
    proposal = json.dumps(_intent("box_at", 1, 4))
    generator_invocations = 0

    def generate() -> str:
        nonlocal generator_invocations
        generator_invocations += 1
        return "{" if generator_invocations == 1 else proposal

    def runtime(limit: int) -> IntentSearchRuntime:
        config = ExperimentConfig(cache_dir=str(tmp_path / "cache"), json_max_retries=1,
                                  limits=Limits(max_requests=limit))
        api = create_inference_api(config, "mock")
        components = create_predicate_components(config, api, context_limit=8192, scale_arm="unscaled")
        return IntentSearchRuntime(
            executor=SokobanIntentExecutor(lambda seed: components.policy_factory(seed, mock=lambda _payload: '{"action":"RIGHT"}')),
            propose=lambda obs, goal, scale: [components.propose(obs, goal, scale, mock=generate)],
            value=lambda obs, goal: components.value(obs, goal, mock=lambda: '{"value":0.5}'),
            metrics=lambda: components.metrics,
        )

    settings = _settings(scales=(None,))
    initial = search_intents(START, TASK_GOAL, mode="full-search", settings=settings, runtime=runtime(5))
    cached = search_intents(START, TASK_GOAL, mode="full-search", settings=settings, runtime=runtime(5))
    assert initial.solved and cached.solved and initial.action_path == cached.action_path == [3, 3]
    assert generator_invocations == 2
    for result in (initial, cached):
        assert result.accounting["logical_generator_calls"] == result.accounting["logical_policy_calls"] == 2
        assert result.accounting["logical_value_calls"] == 1
        assert result.accounting["logical_api_calls"] == 5 and result.accounting["physical_requests"] == 0
    assert cached.accounting["cache_hits"] == 5
    capped = search_intents(START, TASK_GOAL, mode="full-search", settings=settings, runtime=runtime(4))
    assert not capped.solved and capped.finish_reason == "api_budget"
    assert capped.accounting["logical_api_calls"] == capped.accounting["cache_hits"] == 4
    assert capped.accounting["attempted_simulator_actions"] == 1 and not capped.action_path


@pytest.mark.parametrize("mode", ["one-step", "full-search"])
def test_task_completion_is_distinct_from_a_missed_intent(mode: PlannerMode) -> None:
    missed = _intent("player_at", 3, 1)
    runtime, _ = _runtime(_right, lambda _obs, _goal, _scale: [missed])
    result = search_intents(START, TASK_GOAL, mode=mode, runtime=runtime, settings=_settings())
    assert result.solved and result.task_solved and not result.intent_reached
    assert result.action_path == [3, 3]
    assert all(attempt.task_solved and not attempt.intent_reached for attempt in result.attempts)
    assert all(attempt.status == "task_solved" for attempt in result.attempts)
    assert result.endpoint == replay_fragment(START, result.action_path)[-1]


def test_unknown_preservation_rejects_even_when_simulator_reaches_task_goal() -> None:
    near_goal = SokobanSnapshot(("#######", "#-@$.-#", "#--*--#", *START.rows[3:]))
    proposed = _intent("box_at", 1, 4, preserve=[{"predicate": "box_at", "row": 2, "col": 3}])

    def observe(board: Board) -> SokobanObservation:
        rows = to_rows(board)
        if rows[1][2] != "@":
            rows[2] = rows[2][:3] + "?" + rows[2][4:]
        return SokobanObservation(tuple(rows))

    runtime, _ = _runtime(_right, lambda _obs, _goal, _scale: [proposed], observe=observe)
    result = search_intents(near_goal, TASK_GOAL, mode="full-search", runtime=runtime, settings=_settings())
    attempt, = result.attempts
    assert attempt.task_solved and attempt.status == "unknown" and attempt.node is None
    assert not result.solved and not result.action_path
    assert result.accounting["attempted_simulator_actions"] == 1


@pytest.mark.parametrize("mode", ["direct", "one-step", "full-search"])
def test_terminal_root_needs_no_generation_value_or_actions(mode: PlannerMode) -> None:
    solved = replay_fragment(START, [3, 3])[-1]

    def unexpected(*_args: object) -> NoReturn:
        pytest.fail("A terminal root must stop before inference")

    runtime, decisions = _runtime(unexpected, unexpected, value=unexpected)
    result = search_intents(solved, TASK_GOAL, mode=mode, runtime=runtime, settings=_settings())
    assert result.solved and result.task_solved and not result.intent_reached
    assert not result.action_path and not decisions
    assert result.accounting["logical_model_calls"] == 0


def test_fitted_value_uses_only_solved_training_returns(
    tmp_path: Path, cli_files: tuple[list[str], Path, Path, Path],
) -> None:
    _, _, held_out, _ = cli_files
    value = fit_value(tmp_path / "demonstrations.json", discount=0.9)
    assert value.train_trajectories == 1 and value.train_examples == 3
    query = json.loads(held_out.read_text(encoding="utf-8"))
    initial = SokobanSnapshot(tuple(query["observation"]["rows"]))
    value.assert_held_out(initial.observation, query["task_group"])
    assert [value(state.observation, TASK_GOAL) for state in replay_fragment(initial, [3, 3])] == pytest.approx(
        [0.81, 0.9, 1.0], abs=1e-4,
    )
    with pytest.raises(ValueError, match="training layout or task group"):
        value.assert_held_out(initial.observation, "group-train")
    path = tmp_path / "value.json"
    value.save(path)
    assert load_value(path, discount=0.9).identity() == value.identity()
    with pytest.raises(ValueError, match="discount does not match"):
        load_value(path, discount=0.8)


def test_unsupported_task_goal_is_rejected_before_inference() -> None:
    runtime, decisions = _runtime(_right, lambda *_: [])
    with pytest.raises(ValueError, match="all-boxes-on-goals"):
        search_intents(START, "Move without pushing boxes.", mode="direct", runtime=runtime, settings=_settings())
    assert not decisions
