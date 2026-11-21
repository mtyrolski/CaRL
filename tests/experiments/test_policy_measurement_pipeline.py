"""Exercise checked annotation, controller measurement and training together."""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carl.icl.boards import to_rows
from carl.icl.config import Limits, ModelCapabilities, ModelSettings
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.intent_annotations import AnnotationRequest, prepare_annotations
from carl.icl.intent_domain import (
    TASK_GOAL,
    Intent,
    SokobanObservation,
    SokobanSnapshot,
    decode_intent,
    replay_fragment,
)
from carl.icl.intents.generation.data import read_records
from carl.icl.luna_intent_rollout import Action, ActionAttempt, build_prompt
from carl.icl.policy_relative.data import load_fit_data
from carl.icl.policy_relative.measurements import MeasurementSettings, measure, trial_seed
from carl.icl.policy_relative.rollout import ReplayIntentPolicy, run_measurement_attempt
from carl.icl.types import Board


TRAIN = ("#######", "#@$-.-#", "#-$-.-#", "#-----#", "#--#--#", "#-----#", "#######")
VALIDATION = (*TRAIN[:4], "#-##--#", *TRAIN[5:])
CONTROLLER = {"id": "scripted-fixture", "revision": "v1", "deterministic": False}
HISTORY = (ActionAttempt(Action.UP, True),)
CAP = 4


@dataclass(frozen=True)
class CandidateCase:
    predicate: dict[str, str | int]
    witness: tuple[int, ...]
    attempts: int
    success: bool

    def intent(self) -> Intent:
        return decode_intent({"effects": [self.predicate], "preserve": [], "instruction": None})


CASES = {
    "A": CandidateCase({"predicate": "player_at", "row": 2, "col": 2}, (1, 3), 1, True),
    "B": CandidateCase({"predicate": "box_at", "row": 1, "col": 4}, (3, 3), 2, True),
    "C": CandidateCase({"predicate": "player_at", "row": 3, "col": 3}, (1, 3, 1, 3), 4, True),
    "D": CandidateCase({"predicate": "player_at", "row": 3, "col": 1}, (1, 1), 4, False),
}


@dataclass(frozen=True)
class CheckedPool:
    path: Path
    settings: MeasurementSettings
    plans: dict[int, tuple[SokobanSnapshot, tuple[int, ...]]]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _case_name(row: dict[str, Any]) -> str:
    effect, = row["intent"]["effects"]
    return next(name for name, case in CASES.items() if effect == case.predicate)


@pytest.fixture
def checked_pool(tmp_path: Path) -> CheckedPool:
    trajectories = []
    supported: dict[tuple[tuple[str, ...], tuple[str, ...]], Intent] = {}
    for split, board in (("train", TRAIN), ("validation", VALIDATION)):
        snapshot = SokobanSnapshot(board)
        for name, case in CASES.items():
            trajectories.append({
                "source_id": f"{split}-{name}",
                "split": split,
                "start_observation": board,
                "actions": case.witness,
            })
            endpoint = replay_fragment(snapshot, case.witness)[-1]
            supported[board, endpoint.rows] = case.intent()
    source = tmp_path / "demonstrations.json"
    source.write_text(json.dumps({"trajectories": trajectories}), encoding="utf-8")

    def annotate(pair: AnnotationRequest) -> Intent | None:
        return supported.get((pair.start_observation.rows, pair.later_observation.rows))

    directory = tmp_path / "annotations"
    prepare_annotations(source, directory, max_offset=CAP, seed=17, annotator=annotate)
    path = directory / "D_A.jsonl"
    rows = read_records(path)
    assert len(rows) == 8
    counts = {row["record_id"]: CASES[_case_name(row)].attempts for row in rows}
    settings = MeasurementSettings(seed=21, action_cap=CAP, planned_attempts=counts, offset_denominator=8)
    plans = {}
    for row in rows:
        case = CASES[_case_name(row)]
        for index in range(case.attempts):
            actions = case.witness if case.success and index == 0 else (0,) * CAP
            seed = trial_seed(settings.seed, row["record_id"], index)
            plans[seed] = SokobanSnapshot(tuple(row["start_observation"])), actions
    assert len(plans) == 22
    return CheckedPool(path, settings, plans)


class ScriptedPolicy:
    """Use an independent iterator and check the starting prompt each time."""

    def __init__(self, pool: CheckedPool, *, interrupt_after: int | None = None):
        self.pool = pool
        self.interrupt_after = interrupt_after
        self.seeds: list[int] = []

    def __call__(self, seed: int) -> Callable[[str], str]:
        if self.interrupt_after is not None and len(self.seeds) == self.interrupt_after:
            raise KeyboardInterrupt("interrupted fixture")
        self.seeds.append(seed)
        snapshot, plan = self.pool.plans[seed]
        actions = iter(plan)
        first = True

        def decide(prompt: str) -> str:
            nonlocal first
            if first:
                question = json.loads(prompt.rsplit("\n\n", 1)[-1])
                assert tuple(question["state"]) == snapshot.rows
                assert question["attempt_history"] == [{"action": "UP", "no_op": True}]
                assert "source_id" not in question and "later_observation" not in question
                first = False
            return json.dumps({"action": tuple(Action)[next(actions)].value})

        return decide


def test_measurements_reach_the_training_loader_with_probability_weights(tmp_path: Path, checked_pool: CheckedPool):
    policy = ScriptedPolicy(checked_pool)
    output = tmp_path / "calibration"
    manifest = measure(
        checked_pool.path, output, settings=checked_pool.settings, controller=CONTROLLER,
        policy_factory=policy, initial_memory=HISTORY,
    )
    assert manifest["status"] == "complete"
    assert len(policy.seeds) == len(set(policy.seeds)) == 22
    path = output / "D_C.jsonl"
    rows = _jsonl(path)
    assert len(rows) == 8
    trained = load_fit_data(path)
    assert trained.action_cap == CAP and trained.attempts_per_candidate is None
    assert len(trained.train) == len(trained.validation) == 1

    expected = {
        "A": (1, 2, 4 / 7, 3 / 7),
        "B": (1 / 2, 2, 2 / 7, 3 / 7),
        "C": (1 / 4, 4, 1 / 7, 13 / 14),
    }
    for context in (trained.train[0], trained.validation[0]):
        assert {_case_name(candidate.record) for candidate in context.candidates} == set(expected)
        for candidate in context.candidates:
            name = _case_name(candidate.record)
            p, d, w, u = expected[name]
            assert candidate.success_fraction == p
            assert candidate.mean_success_actions == d
            assert candidate.weight == pytest.approx(w)
            assert candidate.u == pytest.approx(u)
            measurement = candidate.record["policy_measurements"]
            assert measurement["planned_attempts"] == CASES[name].attempts
            successful, = [attempt for attempt in measurement["attempts"] if attempt["status"] == "achieved"]
            assert successful["primitive_actions_attempted"] == successful["first_hit_actions"] == d
            arms = measurement["scale_labels"]
            assert arms["unscaled"] is None
            assert arms["absolute_d"] == d
            assert arms["ratio_d_over_ac"] == d / 2
            assert arms["relative_u"] == pytest.approx(u)
            assert arms["offset_rank"] == (3 / 8 if d == 2 else 7 / 8)
            assert arms["normalized_offset"] == d / 8
    failed = [row["policy_measurements"] for row in rows if _case_name(row) == "D"]
    assert len(failed) == 2
    for item in failed:
        assert item["planned_attempts"] == len(item["attempts"]) == 4
        assert item["success_fraction"] == 0
        assert item["mean_success_actions"] is item["weight"] is item["u"] is None
        assert {attempt["status"] for attempt in item["attempts"]} == {"action_limit"}
        assert {attempt["primitive_actions_attempted"] for attempt in item["attempts"]} == {CAP}

    corrupted = tmp_path / "altered.jsonl"
    row = next(row for row in rows if _case_name(row) == "A")
    row["policy_measurements"]["u"] = 0.123
    corrupted.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="u disagrees"):
        load_fit_data(corrupted)


def test_interrupt_and_resume_reuse_completed_attempts_and_freeze_inputs(tmp_path: Path, checked_pool: CheckedPool):
    output = tmp_path / "interrupted"
    interrupted = ScriptedPolicy(checked_pool, interrupt_after=1)
    with pytest.raises(KeyboardInterrupt, match="interrupted fixture"):
        measure(
            checked_pool.path, output, settings=checked_pool.settings, controller=CONTROLLER,
            policy_factory=interrupted, initial_memory=HISTORY,
        )
    saved, = (output / "attempts").glob("*/*.json")
    preserved = saved.read_bytes()
    pool_path = output / "pool.json"
    old_pool = pool_path.read_bytes()
    partial = _jsonl(output / "D_C.jsonl")
    assert all(row["policy_measurements"]["complete"] is False for row in partial)
    assert all(row["policy_measurements"]["weight"] is None for row in partial)
    with pytest.raises(ValueError, match="complete context"):
        load_fit_data(output / "D_C.jsonl")

    followup = ScriptedPolicy(checked_pool)
    changed_pool = json.loads(old_pool)
    changed_pool["candidates"].pop()
    pool_path.write_text(json.dumps(changed_pool), encoding="utf-8")
    with pytest.raises(ValueError, match="changed frozen pool"):
        measure(
            checked_pool.path, output, settings=checked_pool.settings, controller=CONTROLLER,
            policy_factory=followup, initial_memory=HISTORY, resume=True,
        )
    pool_path.write_bytes(old_pool)

    for changed in (replace(checked_pool.settings, action_cap=5), replace(checked_pool.settings, seed=22)):
        with pytest.raises(ValueError):
            measure(
                checked_pool.path, output, settings=changed, controller=CONTROLLER,
                policy_factory=followup, initial_memory=HISTORY, resume=True,
            )
    with pytest.raises(ValueError):
        measure(
            checked_pool.path, output, settings=checked_pool.settings, controller={**CONTROLLER, "revision": "v2"},
            policy_factory=followup, initial_memory=HISTORY, resume=True,
        )
    with pytest.raises(ValueError):
        measure(
            checked_pool.path, output, settings=checked_pool.settings, controller=CONTROLLER,
            policy_factory=followup, initial_memory=(), resume=True,
        )
    assert not followup.seeds
    assert pool_path.read_bytes() == old_pool
    assert saved.read_bytes() == preserved

    manifest = measure(
        checked_pool.path, output, settings=checked_pool.settings, controller=CONTROLLER,
        policy_factory=followup, initial_memory=HISTORY, resume=True,
    )
    assert manifest["status"] == "complete"
    assert set(interrupted.seeds).isdisjoint(followup.seeds)
    assert len(followup.seeds) == 21
    assert len(list((output / "attempts").glob("*/*.json"))) == 22
    assert saved.read_bytes() == preserved
    assert load_fit_data(output / "D_C.jsonl").train


def test_observed_unknown_violation_and_terminal_are_distinct():
    snapshot = SokobanSnapshot(TRAIN)
    maintained = decode_intent({
        "effects": [CASES["A"].predicate],
        "preserve": [{"predicate": "box_at", "row": 1, "col": 2}],
    })
    seen = []

    def observe(board: Board) -> SokobanObservation:
        rows = to_rows(board)
        rows[3] = rows[3][:4] + "?" + rows[3][5:]
        if rows[1][1] != "@":
            rows[1] = rows[1][:2] + "?" + rows[1][3:]
        return SokobanObservation(tuple(rows))

    def decide(prompt: str) -> str:
        seen.append(json.loads(prompt.rsplit("\n\n", 1)[-1]))
        return '{"action":"DOWN"}'

    unresolved = run_measurement_attempt(
        snapshot, maintained, TASK_GOAL, cap=CAP, seed=1, initial_memory=HISTORY,
        policy_factory=lambda _seed: decide, observe=observe,
    )
    assert unresolved["status"] == "unknown"
    assert unresolved["first_hit_actions"] is None
    assert unresolved["primitive_actions_attempted"] == 1
    initial, = seen
    assert initial["state"][3][4] == "?"
    assert initial["attempt_history"] == [{"action": "UP", "no_op": True}]

    violated = run_measurement_attempt(
        snapshot, maintained, TASK_GOAL, cap=CAP, seed=1, initial_memory=HISTORY,
        policy_factory=lambda _seed: lambda _prompt: '{"action":"RIGHT"}',
    )
    assert violated["status"] == "preservation_violated"
    assert violated["first_hit_actions"] is None and violated["task_solved"] is False
    assert violated["primitive_actions_attempted"] == 1

    small = SokobanSnapshot(("#####", "#@$.#", "#---#", "#---#", "#####"))
    goal = decode_intent({"effects": [{"predicate": "player_at", "row": 2, "col": 1}], "preserve": []})
    terminal = run_measurement_attempt(
        small, goal, TASK_GOAL, cap=CAP, seed=1, initial_memory=HISTORY,
        policy_factory=lambda _seed: lambda _prompt: '{"action":"RIGHT"}',
    )
    assert terminal["status"] == "task_solved"
    assert terminal["task_solved"] is True and terminal["first_hit_actions"] is None
    assert terminal["primitive_actions_attempted"] == 1


def test_replay_counts_cached_actions_toward_the_stage_02_logical_limit(tmp_path: Path):
    snapshot = SokobanSnapshot(TRAIN)
    intent = CASES["A"].intent()
    after = replay_fragment(snapshot, [1])[-1]
    prompts = (
        build_prompt(snapshot.restore(), intent, TASK_GOAL, measured=True),
        build_prompt(after.restore(), intent, TASK_GOAL, (ActionAttempt(Action.DOWN, False),), measured=True),
    )
    model = ModelSettings(model="replay-fixture", capabilities=ModelCapabilities(streaming=False))
    cache = tmp_path / "cache"
    wallet = ExperimentAPI(cache, "replay", Limits(max_requests=1))
    replay = ReplayIntentPolicy(wallet, model, 32, 17)
    responses = iter(('{"action":"DOWN"}', '{"action":"RIGHT"}'))

    def fake_response(**_options: Any) -> SimpleNamespace:
        return SimpleNamespace(
            status="completed", output_text=next(responses), output=[], model=model.model,
            id="fixture-response", usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 20, "output_tokens": 1}),
        )

    # Populate the provider cache entirely through an injected local SDK fixture.
    warmed = ExperimentAPI(
        cache, "openai", Limits(max_requests=2),
        client=SimpleNamespace(responses=SimpleNamespace(create=fake_response)),
    )
    for prompt in prompts:
        warmed.call("responses", replay.payload(prompt), transport=model)
    assert warmed.metrics["physical_requests"] == 2
    attempt = run_measurement_attempt(
        snapshot, intent, TASK_GOAL, cap=CAP, seed=17, initial_memory=(),
        policy_factory=lambda _seed: replay,
    )
    assert attempt["status"] == "budget_exhausted"
    assert attempt["primitive_actions_attempted"] == 1
    assert attempt["first_hit_actions"] is None
    assert attempt["model_calls"] == 2
    assert wallet.metrics["cache_hits"] == wallet.metrics["logical_requests"] == 1
    assert wallet.metrics["physical_requests"] == 0 and wallet.client is None
