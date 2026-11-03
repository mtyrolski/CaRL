"""Exercise the verbal judge boundary, budgets and semantic calibration offline."""
from pathlib import Path
import json
from typing import Any, cast

import numpy as np
import pytest

from carl.icl.api import RequestFailed
from carl.icl.boards import from_rows, to_rows
from carl.icl.config import Calibration, ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.types import Board, TrainingData
from scripts.verbal_icl.calibration import calibrate, signature
from scripts.verbal_icl.components import Example, VerbalComponents, describe_transition
from scripts.verbal_icl.search import RolloutFailure, rollout, search
from scripts.verbal_icl.types import Judgment


class FakeComponents:
    """Return scripted judge decisions without simulator successor probing."""

    def __init__(self, reached: bool = False, action: int = 0, *, fail: bool = False) -> None:
        self.reached, self.next_action, self.fail = reached, action, fail
        self.judge_calls = 0

    def propose(self, state: Board, scale: float) -> list[str]:
        return []

    def action(self, state: Board, goal: str, source: Board | None = None) -> int:
        return self.next_action

    def judge(self, state: Board, goal: str, source: Board | None = None) -> Judgment:
        self.judge_calls += 1
        if self.fail:
            raise RequestFailed('synthetic failure')
        return Judgment(self.reached, float(self.reached))

    def value(self, state: Board) -> float:
        return 0.5


def board() -> Board:
    return from_rows(['#####', '#@--#', '#-$-#', '#.--#', '#####'])


def test_judge_controls_stopping_and_noop_actions_use_cap() -> None:
    state = board()
    env = environment(state)
    fake = FakeComponents(reached=False, action=0)
    attempt = rollout(env, fake, state, 'Go somewhere', 3)
    assert len(attempt.actions) == 3 and fake.judge_calls == 3
    assert not attempt.judged_reached and np.array_equal(attempt.endpoint, state)
    fake = FakeComponents(reached=True, action=0)
    attempt = rollout(env, fake, state, 'Go somewhere', 3)
    assert attempt.judged_reached and len(attempt.actions) == 1
    # A possibly wrong judge claim is exposed, never silently replaced by an oracle.
    assert np.array_equal(attempt.endpoint, state)


def test_solved_task_still_judged_after_final_action() -> None:
    state = from_rows(['#####', '#---#', '#@$.#', '#---#', '#####'])
    fake = FakeComponents(action=3)
    attempt = rollout(environment(state), fake, state, 'Push the box', 8)
    assert attempt.solved and not attempt.judged_reached and attempt.actions == [3]
    assert fake.judge_calls == 1


def test_failure_keeps_action_that_preceded_judge_error() -> None:
    state = board()
    fake = FakeComponents(action=3, fail=True)
    with pytest.raises(RolloutFailure) as caught:
        rollout(environment(state), fake, state, 'Go right', 8)
    assert caught.value.attempt.actions == [3]
    assert caught.value.attempt.steps[0].judgment is None


def test_search_accounts_all_fallback_actions_without_exceeding_budget() -> None:
    state = board()
    result = search(environment(state), FakeComponents(), state, (0.8, 0.4), 5, 8, 40)
    assert result.finish_reason == 'action_budget' and result.attempted_actions == 5
    action_count = 0
    for event in result.events:
        actions = event['actions']
        assert isinstance(actions, list)
        action_count += len(actions)
    assert action_count == 5
    assert result.primitive_actions == 5


def test_description_only_refers_to_actual_changed_train_positions() -> None:
    state = board()
    env = environment(state)
    right = np.asarray(env.next_state(state, 3))
    assert describe_transition(to_rows(state), to_rows(right)) == 'Move the player to cell [1,2].'
    with pytest.raises(ValueError, match='does not change'):
        describe_transition(to_rows(state), to_rows(state))


def test_semantic_calibration_uses_its_own_judge_and_tie_midpoint(tmp_path: Path) -> None:
    state = board()
    env = environment(state)
    targets = [np.asarray(env.next_state(state, action)) for action in (1, 3)]
    training: TrainingData = {'contexts': [{'id': 'train-a', 'state': to_rows(state), 'remaining_actions': 3,
        'candidates': [{'subgoal': to_rows(target), 'k': index + 1, 'actions': [action]}
                       for index, (target, action) in enumerate(zip(targets, (1, 3), strict=True))],
        'calibration_indices': [0, 1]}]}
    config = ExperimentConfig(train_boards=1, few_shot=1,
        calibration=Calibration(reference_window=4, candidates=2, sampled_targets=2, trials=1, action_cap=2))
    fake = FakeComponents(reached=True, action=3)
    # Calibration only needs the component's API provider and rollout methods.
    fake.api = type('Api', (), {'provider': 'mock', 'metrics': {}})()  # type: ignore[attr-defined]
    fake.calls = {}  # type: ignore[attr-defined]
    prompts = {'generator': 'generator', 'policy': 'policy', 'judge': 'judge'}
    result = calibrate(config, training, cast(VerbalComponents, fake), prompts, tmp_path)
    assert result['status'] == 'completed' and result['trials'] == 2
    assert [row['target_index'] for row in result['examples']] == [0, 1]
    assert result['contexts_with_hits'] == 1
    assert all(target['u'] == 0.5 and target['p_hat'] == 1 for target in result['targets'])
    assert signature(config, training, prompts, 'mock') != signature(config, training, prompts, 'openai')


def test_model_inputs_use_text_goal_without_future_board_or_bfs_hint() -> None:
    class RecordingAPI:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def call(self, _endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
            del mock
            self.payloads.append(payload)
            role = payload['metadata']['component']
            answer = {'generator': {'subgoals': ['Move player to [1,2].']},
                      'policy': {'action': 'RIGHT'}, 'judge': {'reached': True, 'progress': 0.5, 'reason': 'Player at requested cell'}}[role]
            return {'output_text': json.dumps(answer)}

    state = board()
    next_state = np.asarray(environment(state).next_state(state, 3))
    training: TrainingData = {'contexts': [{'id': 'train-a', 'state': to_rows(state), 'remaining_actions': 3,
        'candidates': [{'subgoal': to_rows(next_state), 'k': 1, 'actions': [3]}], 'calibration_indices': [0]}]}
    config = ExperimentConfig(few_shot=1, num_subgoals=1)
    recorder = RecordingAPI()
    api = cast(ExperimentAPI, recorder)
    reference = LunaComponents(config, api, environment(state), training)
    examples: list[Example] = [{'context': 'train-a', 'target_index': 0, 'source': to_rows(state),
                                'goal': 'Move the player to cell [1,2].', 'u': 0.5}]
    examples.append({**examples[0], 'target_index': 1, 'u': 0.75, 'goal': 'Place a box on goal [3,1].'})
    components = VerbalComponents(config, api, reference, examples,
                                  {'generator': 'generator', 'policy': 'policy', 'judge': 'judge'},
                                  generator_examples=2)
    goals = components.propose(state, 0.5)
    generator_messages = recorder.payloads[0]['input']
    assert len(generator_messages) == 6  # Two examples despite legacy few_shot=1.
    assert json.loads(generator_messages[4]['content'][0]['text'])['subgoals'] == [examples[1]['goal']]
    assert components.action(state, goals[0]) == 3
    assert components.judge(state, goals[0]).reached
    for payload in recorder.payloads:
        query = json.loads(payload['input'][-1]['content'][0]['text'])
        assert 'target' not in query and 'bfs_distance' not in query and 'actions' not in query
        if payload['metadata']['component'] in {'policy', 'judge'}:
            assert query['subgoal'] == goals[0] and isinstance(query['subgoal'], str)


def test_rollout_supplies_original_source_after_each_move() -> None:
    class SourceChecking(FakeComponents):
        def action(self, state: Board, goal: str, source: Board | None = None) -> int:
            assert source is not None and np.array_equal(source, initial)
            return super().action(state, goal, source)

        def judge(self, state: Board, goal: str, source: Board | None = None) -> Judgment:
            assert source is not None and np.array_equal(source, initial)
            return super().judge(state, goal, source)

    initial = board()
    attempt = rollout(environment(initial), SourceChecking(action=3), initial, 'Walk right', 2)
    assert len(attempt.steps) == 2


def test_solved_action_survives_judge_failure() -> None:
    state = from_rows(['#####', '#---#', '#@$.#', '#---#', '#####'])
    with pytest.raises(RolloutFailure) as caught:
        rollout(environment(state), FakeComponents(action=3, fail=True), state, 'Push the box', 8)
    assert caught.value.attempt.solved and caught.value.attempt.steps[-1].solved


def test_placement_goal_uses_box_on_goal_channel() -> None:
    from scripts.verbal_icl.demonstrations import placement_goal
    source = ['#####', '#---#', '#@$.#', '#---#', '#####']
    target = ['#####', '#---#', '#-@*#', '#---#', '#####']
    assert placement_goal(source, target) == 'Place a box on each of these goal squares: [2,3].'
    with pytest.raises(ValueError, match='new goal'):
        placement_goal(source, source)


def test_tiny_profile_constructs_original_reference_offline(tmp_path: Path) -> None:
    from carl.icl.config import ROOT
    from scripts.verbal_icl.runner import configured
    from scripts.verbal_icl.reference import reference_data, build_reference
    config, _prompts, spec = configured(ROOT / 'configs/icl/verbal/tiny.json')
    assert config.locked_policy_fingerprint is None and config.few_shot == 3
    original, training = reference_data(spec, config.model)
    api = ExperimentAPI(tmp_path / 'cache', 'mock', config.limits)
    reference = build_reference(original, training, api)
    assert reference.fingerprint() == original.locked_policy_fingerprint
    assert original.few_shot == 4 and len(training['contexts']) == 8
    assert original.value_examples == 'trajectory'
    assert reference.value_payload(from_rows(training['contexts'][0]['state']))
    assert api.metrics['logical_requests'] == 0
