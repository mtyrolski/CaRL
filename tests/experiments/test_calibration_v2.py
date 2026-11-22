"""Full-state calibration, resume, and verification against the frozen training pool."""
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import carl.icl.calibration as calibration
from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.calibration import _candidate_trials
from carl.icl.calibration import calibrate
from carl.icl.calibration import target_indices
from carl.icl.calibration import weighted_median
from carl.icl.calibration import weighted_percentiles
from carl.icl.config import Calibration
from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment import calibration_path
from carl.icl.experiment import components_for_run
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.search import Attempt
from carl.icl.types import Board
from carl.icl.types import TrainingData


def board_with_player(column: int) -> Board:
    rows = ['########', '#------#', '#---$.-#', '#------#', '#------#', '#------#', '#------#', '########']
    row = list(rows[1])
    row[column] = '@'
    rows[1] = ''.join(row)
    return from_rows(rows)


def make_training(source: Board, targets: list[Board]) -> TrainingData:
    return cast(TrainingData, {'contexts': [{'id': 'train', 'state': to_rows(source), 'remaining_actions': 12,
        'candidates': [{'subgoal': to_rows(target), 'k': index + 1, 'actions': [3] * (index + 1)}
                       for index, target in enumerate(targets)],
        'calibration_indices': list(range(len(targets)))}]})


def make_components(tmp_path: Path, config: ExperimentConfig, training: TrainingData) -> LunaComponents:
    board = from_rows(training['contexts'][0]['state'])
    return LunaComponents(config, ExperimentAPI(tmp_path, 'mock', config.calibration.limits), environment(board), training)


@pytest.fixture
def prepared_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic_token_estimates: None,
) -> tuple[ExperimentConfig, TrainingData, Board, dict]:
    source = board_with_player(1)
    target = np.asarray(environment(source).next_state(source, 3))
    training = make_training(source, [target])
    config = ExperimentConfig(
        method='relative_u', few_shot=1, data_dir=str(tmp_path / 'data'), cache_dir=str(tmp_path / 'cache'),
        calibration=Calibration(trials=2, action_cap=2, sampled_targets=1),
    )
    components = make_components(config.path('cache_dir'), config, training)
    monkeypatch.setattr(components, 'probabilities', lambda _state, _target: np.array([0., 0., 0., 1.]))
    result = calibrate(config, components, config.path('data_dir') / 'calibration')
    assert result['status'] == 'completed' and len(result['examples']) == 1
    return config, training, source, result


def test_valid_calibration_is_accepted_without_api_calls(
    prepared_calibration: tuple[ExperimentConfig, TrainingData, Board, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, training, board, result = prepared_calibration
    monkeypatch.setattr(ExperimentAPI, 'call', lambda *_args, **_kwargs: pytest.fail('Unexpected inference call'))
    components = components_for_run(config, 'mock', training, board)
    assert components.calibrated == result
    assert components.api.metrics['requests'] == components.api.metrics['logical_requests'] == 0


@pytest.mark.parametrize(('field', 'invalid', 'message'), [
    ('signature', 'f' * 64, 'signature'),
    ('state', to_rows(board_with_player(3)), 'differs from its training trajectory'),
    ('subgoal', to_rows(board_with_player(3)), 'differs from its training trajectory'),
    ('context', 'foreign-context', 'unknown training context'),
    ('target_index', 99, 'unselected training target'),
    ('actions', [2], 'differs from its training trajectory'),
    ('k', 2, 'differs from its training trajectory'),
    ('u', 0.1, 'integrity check'),
    ('weight', 0.3, 'integrity check'),
    ('a_c', 100.0, 'integrity check'),
], ids=['signature-signature', 'state-differs from its training trajectory',
        'subgoal-differs from its training trajectory', 'context-unknown training context',
        'candidate_index-unselected training target', 'actions-differs from its training trajectory',
        'offset-differs from its training trajectory', 'label-integrity check',
        'weight-integrity check', 'median-integrity check'])
def test_tampered_calibration_is_rejected_before_inference(
    prepared_calibration: tuple[ExperimentConfig, TrainingData, Board, dict], monkeypatch: pytest.MonkeyPatch,
    field: str, invalid: Any, message: str,
) -> None:
    config, training, board, result = prepared_calibration
    changed = result if field == 'signature' else result['examples'][0]
    changed[field] = invalid
    api = ExperimentAPI(config.path('cache_dir'), 'mock', config.limits)
    reader = LunaComponents(config, api, environment(board), training)
    write_json(calibration_path(config, reader), result)
    monkeypatch.setattr(ExperimentAPI, 'call', lambda *_args, **_kwargs: pytest.fail('Unexpected inference call'))
    with pytest.raises(ValueError, match=message):
        components_for_run(config, 'mock', training, board)
    assert api.metrics['requests'] == 0


def test_manual_unequal_weights_ties_and_weighted_median() -> None:
    # R=4: p = (2/4, 1/4, 3/4, 0/4). Exclude the zero-hit target.
    # Normalized weights are (1/3, 1/6, 1/2), with a shared d=2 tie of mass 1/2.
    weights, percentiles = weighted_percentiles([2, 2, 4], [2, 1, 3])
    np.testing.assert_allclose(weights, [1 / 3, 1 / 6, 1 / 2])
    np.testing.assert_allclose(percentiles, [1 / 4, 1 / 4, 3 / 4])
    assert weighted_median([2, 2, 4], weights.tolist()) == 2
    assert weighted_median([3, 1, 2], [0.6, 0.1, 0.3]) == 3
    # Exact half (6/12) must select 3 even if binary floating point rounds below 0.5.
    weights, _ = weighted_percentiles([1, 2, 3, 4, 5], [2, 3, 1, 3, 3])
    assert weighted_median([1, 2, 3, 4, 5], weights.tolist()) == 3
    np.testing.assert_allclose(weighted_percentiles([3, 3], [1, 3])[1], [0.5, 0.5])
    with pytest.raises(ValueError):
        weighted_percentiles([2, 4], [0, 1])


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_calibration_excludes_zero_hits_and_preserves_target_statistics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = board_with_player(1)
    targets = [board_with_player(index) for index in (2, 3, 4, 5)]
    training = make_training(source, targets)
    config = ExperimentConfig(few_shot=1, calibration=Calibration(trials=4, action_cap=8))
    components = make_components(tmp_path, config, training)
    outcome_map = {state_hash(target): durations for target, durations in zip(
        targets, ([2, 2, None, None], [2, None, None, None], [4, 4, 4, None], [None] * 4), strict=True)}
    counts = dict.fromkeys(outcome_map, 0)

    def fake_rollout(_env: Any, _policy: Any, _source: Board, target: Board, *, cap: int, rng: Any,
                     on_step: Any = None, on_action: Any = None) -> Attempt:
        del cap, rng
        digest = state_hash(target)
        trial = counts[digest]
        counts[digest] += 1
        length = outcome_map[digest][trial]
        # A terminal state that is not the target must remain an exact-target failure.
        solved_early = target is not None and digest == state_hash(targets[-1])
        actions = [3] * (length if length is not None else 1)
        for action in actions:
            on_step()
            on_action(_source, action, target)
        return Attempt(target if length is not None else _source, actions, length is not None, solved_early)

    monkeypatch.setattr(calibration, 'rollout', fake_rollout)
    result = calibrate(config, components, tmp_path / 'labels')
    assert result['status'] == 'completed'
    assert result['trials'] == 16 and result['excluded_targets'] == 1 and result['excluded_contexts'] == 0
    assert result['single_time_contexts'] == 0 and result['multi_time_contexts'] == 1
    rows = sorted(result['examples'], key=lambda item: item['k'])
    assert [row['hits'] for row in rows] == [2, 1, 3]
    np.testing.assert_allclose([row['p_hat'] for row in rows], [0.5, 0.25, 0.75])
    np.testing.assert_allclose([row['weight'] for row in rows], [1 / 3, 1 / 6, 1 / 2])
    np.testing.assert_allclose([row['u'] for row in rows], [0.25, 0.25, 0.75])
    np.testing.assert_allclose([row['cdf_before'] for row in rows], [0, 0, 0.5])
    np.testing.assert_allclose([row['cdf_at'] for row in rows], [0.5, 0.5, 1])
    assert [row['a_c'] for row in rows] == [2, 2, 2]
    np.testing.assert_allclose([row['ratio_d_over_ac'] for row in rows], [1, 1, 2])
    path = tmp_path / 'labels' / result['signature']
    all_targets = json.loads((path / 'calibration_targets.json').read_text())
    all_trials = json.loads((path / 'calibration_trials.json').read_text())
    assert len(all_targets) == 4 and len(all_trials) == 16
    assert all_targets[-1]['excluded_reason'] == 'zero_exact_hits'
    assert all(row['solved_before_target'] and not row['exact_hit'] for row in all_trials if row['k'] == 4)


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_real_rollout_terminal_before_target_is_not_a_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = from_rows(['#####', '#---#', '#@$.#', '#---#', '#####'])
    env = environment(source)
    target = np.asarray(env.next_state(source, 0))  # Walk up; unrelated to solving by pushing right.
    training = make_training(source, [target])
    config = ExperimentConfig(few_shot=1, calibration=Calibration(trials=3, action_cap=2))
    components = make_components(tmp_path, config, training)
    monkeypatch.setattr(components, 'probabilities', lambda _state, _target: np.array([0., 0., 0., 1.]))
    result = calibrate(config, components, tmp_path / 'labels')
    assert result['status'] == 'completed' and result['examples'] == []
    assert result['excluded_targets'] == result['excluded_contexts'] == 1
    trials = json.loads((tmp_path / 'labels' / result['signature'] / 'calibration_trials.json').read_text())
    assert len(trials) == 3 and all(row['task_solved'] and row['solved_before_target'] for row in trials)
    assert all(not row['exact_hit'] and row['first_hit'] is None for row in trials)


def test_distinct_candidates_are_sampled_reproducibly_without_replacement() -> None:
    source = board_with_player(1)
    one, two, three = (board_with_player(i) for i in (2, 3, 4))
    targets = [one, one, source, two, three, board_with_player(5)]
    training = make_training(source, targets)
    context = training['contexts'][0]
    config = ExperimentConfig(seed=321, calibration=Calibration(reference_window=5, sampled_targets=3))
    indices, entropy = target_indices(config, context, 0)
    assert (indices, entropy) == target_indices(config, context, 0)
    assert len(indices) == len(set(indices)) == 3
    assert all(context['candidates'][i]['k'] <= 5 for i in indices)
    digests = [state_hash(from_rows(context['candidates'][i]['subgoal'])) for i in indices]
    assert len(set(digests)) == 3 and state_hash(source) not in digests


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_cache_reuses_policy_distribution_but_not_sampled_rollout(tmp_path: Path) -> None:
    source = board_with_player(1)
    target = board_with_player(2)
    training = make_training(source, [target])
    config = ExperimentConfig(seed=112, few_shot=1, calibration=Calibration(trials=4, action_cap=1))
    components = make_components(tmp_path, config, training)
    directory = tmp_path / 'trials'
    counted: list[int] = []
    data = _candidate_trials(config, components, training['contexts'][0], 0,
                             context_index=0, directory=directory, on_step=lambda: counted.append(1))
    assert len(data['records']) == len(counted) == 4
    assert components.api.metrics['requests'] == 1 and components.api.metrics['cache_hits'] == 3
    assert len({tuple(item['seed_entropy']) for item in data['records']}) == 4
    # Restoring completed trials performs no new draws and creates no extra observations.
    restored = _candidate_trials(config, components, training['contexts'][0], 0,
                                 context_index=0, directory=directory, on_step=lambda: counted.append(1))
    assert restored == data and len(counted) == 4
    assert components.api.metrics['logical_requests'] == 4
    distribution = components.probabilities(source, target)
    for record in data['records']:
        rng = np.random.default_rng(np.random.SeedSequence(record['seed_entropy']))
        assert record['actions'] == [int(rng.choice(4, p=distribution))]


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_exact_target_hit_can_also_solve_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = from_rows(['#####', '#---#', '#@$.#', '#---#', '#####'])
    target = np.asarray(environment(source).next_state(source, 3))
    training = make_training(source, [target])
    config = ExperimentConfig(few_shot=1, calibration=Calibration(trials=2, action_cap=2))
    components = make_components(tmp_path, config, training)
    monkeypatch.setattr(components, 'probabilities', lambda _state, _target: np.array([0., 0., 0., 1.]))
    result = calibrate(config, components, tmp_path / 'labels')
    assert result['examples'][0]['hits'] == 2 and result['examples'][0]['d'] == 1
    assert result['single_time_contexts'] == 1 and result['multi_time_contexts'] == 0
    assert result['examples'][0]['u'] == 0.5
    trials = json.loads((tmp_path / 'labels' / result['signature'] / 'calibration_trials.json').read_text())
    assert all(row['exact_hit'] and row['task_solved'] and not row['solved_before_target'] for row in trials)


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_interrupted_target_recovery_and_resume_do_not_double_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from carl.icl.api import BudgetExceeded

    source, target = board_with_player(1), board_with_player(3)
    training = make_training(source, [target])
    training['contexts'][0]['candidates'][0].update(k=2, actions=[3, 3])
    config = ExperimentConfig(few_shot=1, calibration=Calibration(trials=3, action_cap=2))
    components = make_components(tmp_path, config, training)
    calls: list[int] = []

    def interrupted_policy(_state: Board, _target: Board) -> np.ndarray:
        calls.append(1)
        if len(calls) == 6:
            raise BudgetExceeded('local test interruption')
        return np.array([0., 0., 0., 1.])

    monkeypatch.setattr(components, 'probabilities', interrupted_policy)
    result = calibrate(config, components, tmp_path / 'labels')
    path = tmp_path / 'labels' / result['signature']
    trials = json.loads((path / 'calibration_trials.json').read_text())
    targets = json.loads((path / 'calibration_targets.json').read_text())
    assert result['status'] == 'partial' and result['trials'] == 2 and len(trials) == 2
    assert result['calibration_actions'] == 4 and result['total_executed_actions'] == 5
    assert len(targets) == 1 and targets[0]['excluded_reason'] == 'incomplete'
    assert targets[0]['hits'] == 2 and targets[0]['interrupted'][0]['actions'] == [3]
    monkeypatch.setattr(components, 'probabilities', lambda _state, _target: np.array([0., 0., 0., 1.]))
    resumed = calibrate(config, components, tmp_path / 'labels')
    trials = json.loads((path / 'calibration_trials.json').read_text())
    assert resumed['status'] == 'completed' and resumed['trials'] == len(trials) == 3
    assert resumed['calibration_actions'] == 6 and resumed['total_executed_actions'] == 7
    assert len({tuple(row['seed_entropy']) for row in trials}) == 3
    assert len(resumed['invocations']) == 2
    assert calibrate(config, components, tmp_path / 'labels') == resumed


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_calibration_signature_tracks_policy_sampling_and_data(tmp_path: Path) -> None:
    from carl.icl.calibration import calibration_id

    source = board_with_player(1)
    training = make_training(source, [board_with_player(2)])
    config = ExperimentConfig(seed=1, few_shot=1, calibration=Calibration(trials=3, action_cap=2, sampled_targets=1))
    base = make_components(tmp_path, config, training)
    fingerprint = calibration_id(config, base)
    for changed in (config.model_copy(update={'seed': 2}),
                    config.model_copy(update={'model': 'other-model'}),
                    config.model_copy(update={'policy_temperature': 0.5}),
                    config.model_copy(update={'calibration': config.calibration.model_copy(update={'trials': 4})}),
                    config.model_copy(update={'calibration': config.calibration.model_copy(update={'action_cap': 3})}),
                    config.model_copy(update={'calibration': config.calibration.model_copy(update={'sampled_targets': 2})})):
        assert calibration_id(changed, make_components(tmp_path, changed, training)) != fingerprint
    changed_training = make_training(source, [board_with_player(3)])
    assert calibration_id(config, make_components(tmp_path, config, changed_training)) != fingerprint
    # Only accounting limits change: calibrated labels and sampling stay identical.
    changed_limits = config.calibration.limits.model_copy(update={'max_requests': 16})
    accounting = config.model_copy(update={'calibration': config.calibration.model_copy(update={'limits': changed_limits})})
    assert calibration_id(accounting, make_components(tmp_path, accounting, training)) == fingerprint


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_concurrent_calibration_is_rejected_before_trials(tmp_path: Path) -> None:
    from filelock import FileLock

    from carl.icl.api import BudgetExceeded
    from carl.icl.calibration import calibration_id

    source, target = board_with_player(1), board_with_player(2)
    training = make_training(source, [target])
    config = ExperimentConfig(few_shot=1, calibration=Calibration(trials=1, action_cap=1))
    components = make_components(tmp_path, config, training)
    root = tmp_path / 'labels'
    folder = root / calibration_id(config, components)
    folder.mkdir(parents=True)
    with FileLock(str(folder / '.calibration.lock')):
        with pytest.raises(BudgetExceeded, match='already running'):
            calibrate(config, components, root)
    assert not list(folder.glob('context_*'))
