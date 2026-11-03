"""Policy-relative calibration on distinct verified training-trajectory targets."""
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from filelock import FileLock, Timeout

from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows, state_hash
from carl.icl.config import ExperimentConfig
from carl.icl.luna_components.formatting import task
from carl.icl.search import rollout
from carl.icl.types import Board, JSONObject, TrainingContext

if TYPE_CHECKING:
    from carl.icl.luna import LunaComponents

CALIBRATION_VERSION = 'policy-relative-exact-v2'


def weighted_percentiles(times: list[float], hits: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Assign tied successful times the midpoint of their shared weighted CDF interval."""
    times_array, counts = np.asarray(times, float), np.asarray(hits, float)
    if (len(times_array) == 0 or len(times_array) != len(counts)
            or not np.isfinite(times_array).all() or not np.isfinite(counts).all() or np.any(counts <= 0)):
        raise ValueError('Only finite successful first-hit times and positive hit counts are labelable')
    weights = counts / counts.sum()
    percentiles = np.empty_like(times_array)
    unique_times = np.unique(times_array)
    if len(unique_times) == 1:
        percentiles.fill(0.5)
        return weights, percentiles
    lower = 0.0
    for time in unique_times:
        mask = times_array == time
        mass = float(weights[mask].sum())
        percentiles[mask] = lower + mass / 2
        lower += mass
    return weights, percentiles


def weighted_median(times: list[float], weights: list[float]) -> float:
    """Return the lower weighted median: first time whose CDF reaches one half."""
    if not times or len(times) != len(weights):
        raise ValueError('Expected finite times and positive weights')
    order = np.argsort(np.asarray(times, dtype=float), kind='stable')
    values, mass = np.asarray(times, dtype=float)[order], np.asarray(weights, dtype=float)[order]
    if not np.isfinite(values).all() or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError('Expected finite times and positive weights')
    threshold = float(mass.sum()) / 2
    cumulative = np.cumsum(mass)
    tolerance = np.finfo(float).eps * abs(threshold) * max(len(mass), 1)
    return float(values[min(int(np.searchsorted(cumulative, threshold - tolerance, side='left')), len(values) - 1)])


def calibration_id(config: ExperimentConfig, components: "LunaComponents") -> str:
    """Bind model, training content, policy, seeded sampling, window, repetitions and cap."""
    specification = {'version': CALIBRATION_VERSION, 'policy': components.fingerprint(),
                     'training_sha256': hashlib.sha256(json.dumps(components.training, sort_keys=True).encode()).hexdigest(),
                     'calibration': config.calibration.model_dump(exclude={'limits'}), 'seed': config.seed,
                     'target_sampling': 'uniform-distinct-without-replacement-v1',
                     'rollout_sampling': 'numpy-seedsequence-pcg64-categorical-v1', 'numpy_version': np.__version__,
                     'provider': 'mock' if components.api.provider == 'mock' else 'openai'}
    return hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()


def calibration_result_digest(summary: JSONObject) -> str:
    """Detect accidental modification of labels, weights, medians or selected IDs."""
    payload = {key: summary.get(key) for key in ('signature', 'policy_fingerprint', 'examples', 'contexts', 'selected_example_ids')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def target_indices(config: ExperimentConfig, context: TrainingContext, context_index: int) -> tuple[list[int], list[int]]:
    """Sample unique non-source states without replacement from the reference window."""
    unique = {state_hash(from_rows(context['state']))}
    eligible = []
    for index, candidate in enumerate(context['candidates']):
        if not 1 <= candidate['k'] <= config.calibration.reference_window:
            continue
        digest = state_hash(from_rows(candidate['subgoal']))
        if digest not in unique:
            eligible.append(index)
            unique.add(digest)
    entropy = [config.seed, context_index, int(hashlib.sha256(context['id'].encode()).hexdigest()[:8], 16)]
    count = config.calibration.sampled_targets
    if count is None:
        # Legacy prepared datasets already contain a uniform draw without replacement.
        selected = [index for index in context['calibration_indices'] if index in eligible]
    else:
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        selected = [int(index) for index in rng.choice(eligible, min(count, len(eligible)), replace=False)]
    return selected, entropy


def _action_recorder(actions: list[int]) -> Callable[[Board, int, Board], None]:
    def record(_before: Board, action: int, _after: Board) -> None:
        actions.append(action)

    return record


def _candidate_trials(config: ExperimentConfig, components: "LunaComponents", context: TrainingContext,
                      candidate_index: int, *, context_index: int, directory: Path,
                      on_step: Callable[[], None]) -> JSONObject:
    """Persist each independent action-sampling stream; cache only policy distributions."""
    path = directory / f'context_{context_index}_candidate_{candidate_index}.json'
    data: JSONObject = json.loads(path.read_text()) if path.exists() else {'records': [], 'interrupted': []}
    candidate = context['candidates'][candidate_index]
    for trial in range(len(data['records']), config.calibration.trials):
        entropy = [config.seed, context_index, candidate_index, trial]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        actions: list[int] = []
        cache_before = int(components.api.metrics['cache_hits'])

        try:
            outcome = rollout(components.env, components, from_rows(context['state']), from_rows(candidate['subgoal']),
                              cap=config.calibration.action_cap, rng=rng, on_step=on_step,
                              on_action=_action_recorder(actions))
        except (BudgetExceeded, RequestFailed) as exc:
            data['interrupted'].append({'trial': trial, 'seed_entropy': entropy, 'actions': actions,
                                        'error_type': type(exc).__name__, 'error': str(exc)})
            write_json(path, data)
            raise
        data['records'].append({'trial': trial, 'seed_entropy': entropy, 'rng': 'numpy.SeedSequence + PCG64',
                                'actions': outcome.actions, 'attempted_actions': len(outcome.actions),
                                'exact_hit': bool(outcome.reached), 'first_hit': len(outcome.actions) if outcome.reached else None,
                                'task_solved': bool(outcome.solved), 'solved_before_target': bool(outcome.solved and not outcome.reached),
                                'endpoint': state_hash(outcome.state),
                                'local_policy_cache_hits': int(components.api.metrics['cache_hits']) - cache_before})
        write_json(path, data)
    return data


def calibrate(config: ExperimentConfig, components: "LunaComponents", directory: Path) -> JSONObject:
    """Prevent concurrent runs from counting or overwriting the same trial twice."""
    lock_dir = directory / calibration_id(config, components)
    lock_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock_dir / '.calibration.lock'), timeout=0):
            return _calibrate_locked(config, components, directory)
    except Timeout as exc:
        raise BudgetExceeded('This calibration is already running in another process') from exc


def _calibrate_locked(config: ExperimentConfig, components: "LunaComponents", directory: Path) -> JSONObject:
    """Run R seeded rollouts per distinct target; retain exclusions and durable trials."""
    signature = calibration_id(config, components)
    directory = directory / signature
    summary_path = directory / 'calibration.json'
    history = []
    if summary_path.exists():
        old: JSONObject = json.loads(summary_path.read_text())
        if old['status'] == 'completed':
            return old
        history = old.get('invocations', [])
    summary: JSONObject = {'status': 'partial', 'signature': signature, 'version': CALIBRATION_VERSION,
        'policy_fingerprint': components.fingerprint(), 'provider': 'mock' if components.api.provider == 'mock' else 'openai',
        'examples': [], 'excluded_targets': 0, 'excluded_contexts': 0, 'contexts_completed': 0,
        'calibration_actions': 0, 'trials': 0, 'config': config.calibration.model_dump(),
        'context_count': len(components.training['contexts']), 'target_sampling_seed': config.seed,
        'randomness_note': 'Policy distributions are cached by full input; each rollout independently resamples '
                           'categorical actions with its own SeedSequence. No rollout is replayed as a new observation. '
                           'Estimates are conditional on the realized cached policy distributions; provider model draws '
                           'have no controlled seed and cannot be reproduced without the response cache.'}
    invocation_actions = 0
    targets: list[JSONObject] = []
    trial_rows: list[JSONObject] = []
    context_rows: list[JSONObject] = []

    def count_step() -> None:
        nonlocal invocation_actions
        invocation_actions += 1

    try:
        for ci, context in enumerate(components.training['contexts']):
            indices, target_seed = target_indices(config, context, ci)
            retained: list[JSONObject] = []
            for index in indices:
                candidate = context['candidates'][index]
                data = _candidate_trials(config, components, context, index, context_index=ci,
                                         directory=directory, on_step=count_step)
                records = data['records']
                for record in records:
                    trial_rows.append({'context': context['id'], 'target_index': index, 'k': candidate['k'], **record})
                summary['calibration_actions'] += sum(record['attempted_actions'] for record in records)
                summary['trials'] += len(records)
                times = [record['first_hit'] for record in records if record['exact_hit']]
                target: JSONObject = {'context': context['id'], 'target_index': index, 'k': candidate['k'],
                    'state': context['state'], 'goal': task(from_rows(context['state'])),
                    'subgoal': candidate['subgoal'], 'actions': candidate['actions'],
                    'target_hash': state_hash(from_rows(candidate['subgoal'])), 'target_sampling_entropy': target_seed,
                    'trials': len(records), 'hits': len(times), 'p_hat': len(times) / config.calibration.trials,
                    'success_rate': len(times) / config.calibration.trials, 'd': float(np.mean(times)) if times else None,
                    'excluded_reason': None if times else 'zero_exact_hits'}
                targets.append(target)
                if times:
                    retained.append(target)
                else:
                    summary['excluded_targets'] += 1
            if retained:
                weights, percentiles = weighted_percentiles([row['d'] for row in retained], [row['hits'] for row in retained])
                median = weighted_median([row['d'] for row in retained], [float(row['hits']) for row in retained])
                for row, weight, percentile in zip(retained, weights, percentiles, strict=True):
                    lower = sum(float(w) for other, w in zip(retained, weights, strict=True) if other['d'] < row['d'])
                    upper = sum(float(w) for other, w in zip(retained, weights, strict=True) if other['d'] <= row['d'])
                    row.update(weight=float(weight), cdf_before=lower, cdf_at=upper, u=float(percentile), a_c=median,
                               ratio_d_over_ac=row['d'] / median, normalized_offset=row['k'] / config.offset_denominator)
                summary['examples'].extend(retained)
                context_rows.append({'context': context['id'], 'a_c': median, 'retained': len(retained),
                                     'distinct_d': len({row['d'] for row in retained}), 'excluded': False})
            else:
                summary['excluded_contexts'] += 1
                context_rows.append({'context': context['id'], 'a_c': None, 'retained': 0, 'distinct_d': 0, 'excluded': True})
            summary['contexts_completed'] += 1
        summary['status'] = 'completed'
    except (BudgetExceeded, RequestFailed) as exc:
        summary['error'] = str(exc)
    # Reconstruct the aggregate from durable per-target files, including the
    # partially completed target if the run was interrupted mid-context.
    seen_trials = {(row['context'], row['target_index'], row['trial']) for row in trial_rows}
    seen_targets = {(row['context'], row['target_index']) for row in targets}
    for ci, context in enumerate(components.training['contexts']):
        indices, entropy = target_indices(config, context, ci)
        for index in indices:
            path = directory / f'context_{ci}_candidate_{index}.json'
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            candidate = context['candidates'][index]
            for record in data['records']:
                key = (context['id'], index, record['trial'])
                if key not in seen_trials:
                    trial_rows.append({'context': context['id'], 'target_index': index, 'k': candidate['k'], **record})
            if (context['id'], index) not in seen_targets:
                records = data['records']
                times = [record['first_hit'] for record in records if record['exact_hit']]
                targets.append({'context': context['id'], 'target_index': index, 'k': candidate['k'],
                                'state': context['state'], 'goal': task(from_rows(context['state'])),
                                'subgoal': candidate['subgoal'], 'target_hash': state_hash(from_rows(candidate['subgoal'])),
                                'target_sampling_entropy': entropy, 'trials': len(records), 'hits': len(times),
                                'p_hat': None, 'd': None, 'excluded_reason': 'incomplete',
                                'interrupted': data.get('interrupted', [])})
    summary['trials'] = len(trial_rows)
    summary['calibration_actions'] = sum(row['attempted_actions'] for row in trial_rows)
    if summary['status'] == 'completed' and summary['examples']:
        count = len({row['context'] for row in summary['examples']})
        selected = components._diverse(summary['examples'], count)  # pylint: disable=protected-access
        chosen_ids = []
        for example in selected:
            target_index = example.get('target_index')
            if target_index is None:
                raise ValueError('Calibrated example lacks its target index')
            chosen_ids.append({'context': example['context'], 'target_index': target_index})
        summary['selected_example_ids'] = chosen_ids
    summary['contexts'] = context_rows
    summary['single_time_contexts'] = sum(row['distinct_d'] == 1 for row in context_rows)
    summary['multi_time_contexts'] = sum(row['distinct_d'] >= 2 for row in context_rows)
    summary['api'] = dict(components.api.metrics)
    summary['invocations'] = history + [{'attempted_actions': invocation_actions, 'api': dict(components.api.metrics)}]
    summary['total_executed_actions'] = sum(item['attempted_actions'] for item in summary['invocations'])
    if summary['status'] == 'completed':
        summary['result_digest'] = calibration_result_digest(summary)
    write_json(directory / 'calibration_trials.json', trial_rows)
    write_json(directory / 'calibration_targets.json', targets)
    write_json(summary_path, summary)
    return summary
