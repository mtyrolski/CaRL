"""Recompute completed calibration labels from durable first-hit trials, offline."""
import argparse
import csv
import json
from collections import defaultdict
from typing import Any

import numpy as np

from carl.icl.api import write_json
from carl.icl.calibration import calibration_result_digest, weighted_median, weighted_percentiles
from carl.icl.config import load_config
from carl.icl.experiment import calibration_components, calibration_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/icl/campaign_data/calibration_live_v2.json')
    parser.add_argument('--provider', choices=('replay', 'mock'), default='replay')
    args = parser.parse_args()
    config = load_config(args.config)
    # Replay selects the live calibration signature without sending any request.
    components = calibration_components(config, args.provider)
    directory = calibration_path(config, components).parent
    summary_path = directory / 'calibration.json'
    targets_path = directory / 'calibration_targets.json'
    summary = json.loads(summary_path.read_text())
    targets: list[dict[str, Any]] = json.loads(targets_path.read_text())
    trials = json.loads((directory / 'calibration_trials.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError('Only completed calibration can be audited and relabeled')
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[(trial['context'], trial['target_index'])].append(trial)
    if len(grouped) != len(targets) or len(trials) != len(targets) * config.calibration.trials:
        raise ValueError('Expected R completed trials for every selected target')
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_fields = ('p_hat', 'd', 'weight', 'cdf_before', 'cdf_at', 'u', 'a_c', 'ratio_d_over_ac', 'normalized_offset')
    original = {(row['context'], row['target_index']): {key: row.get(key) for key in label_fields} for row in targets}
    context_indices = {context['id']: index for index, context in enumerate(components.training['contexts'])}
    for target in targets:
        ci = context_indices[target['context']]
        checkpoint = directory / f"context_{ci}_candidate_{target['target_index']}.json"
        target['interrupted'] = json.loads(checkpoint.read_text()).get('interrupted', [])
        key = (target['context'], target['target_index'])
        observations = grouped[key]
        if len(observations) != config.calibration.trials or len({row['trial'] for row in observations}) != len(observations):
            raise ValueError('Duplicate or missing trial index')
        hits = [row['first_hit'] for row in observations if row['exact_hit']]
        if len(hits) != target['hits'] or target['trials'] != len(observations):
            raise ValueError('Target counts disagree with saved trials')
        if any(row['exact_hit'] != (row['first_hit'] is not None) for row in observations):
            raise ValueError('Inconsistent exact-hit and first-hit event')
        if any(row['attempted_actions'] != len(row['actions']) or
               (row['exact_hit'] and row['first_hit'] != len(row['actions'])) or
               row['solved_before_target'] != (row['task_solved'] and not row['exact_hit']) for row in observations):
            raise ValueError('Inconsistent rollout length or task-terminal accounting')
        if hits:
            target['p_hat'] = len(hits) / config.calibration.trials
            target['success_rate'] = target['p_hat']
            target['d'] = float(np.mean(hits))
            by_context[target['context']].append(target)
        elif target['excluded_reason'] != 'zero_exact_hits':
            raise ValueError('Zero-hit target has inconsistent exclusion')
    previous = {row['context']: row['a_c'] for row in summary['contexts'] if not row['excluded']}
    examples = []
    changed = []
    for context in summary['contexts']:
        rows = by_context.get(context['context'], [])
        if not rows:
            continue
        weights, percentiles = weighted_percentiles([row['d'] for row in rows], [row['hits'] for row in rows])
        median = weighted_median([row['d'] for row in rows], [float(row['hits']) for row in rows])
        if previous[context['context']] != median:
            changed.append({'context': context['context'], 'previous_a_c': previous[context['context']], 'a_c': median})
        context['a_c'] = median
        for row, weight, percentile in zip(rows, weights, percentiles, strict=True):
            before = sum(float(w) for other, w in zip(rows, weights, strict=True) if other['d'] < row['d'])
            at = sum(float(w) for other, w in zip(rows, weights, strict=True) if other['d'] <= row['d'])
            row.update(weight=float(weight), cdf_before=before, cdf_at=at, u=float(percentile), a_c=median,
                       ratio_d_over_ac=row['d'] / median, normalized_offset=row['k'] / config.offset_denominator)
        examples.extend(rows)
    selected = components._diverse(examples, len(by_context)) if examples else []  # pylint: disable=protected-access
    ids = [{'context': row['context'], 'target_index': row.get('target_index')} for row in selected]
    if ids != summary.get('selected_example_ids'):
        raise ValueError('Relabeling unexpectedly changed weighted demonstration selection')
    if changed:
        write_json(directory / 'calibration_before_median_audit.json', summary)
        write_json(directory / 'calibration_targets_before_median_audit.json', json.loads(targets_path.read_text()))
    corrected_fields = []
    for row in examples:
        before = original[(row['context'], row['target_index'])]
        for field in label_fields:
            previous_value, current_value = before[field], row.get(field)
            if previous_value is not None and current_value is not None and not np.isclose(previous_value, current_value, rtol=1e-12, atol=1e-12):
                corrected_fields.append({'context': row['context'], 'target_index': row['target_index'],
                                         'field': field, 'before': previous_value, 'after': current_value})
    summary['examples'] = examples
    summary['interrupted_attempts'] = sum(len(target['interrupted']) for target in targets)
    summary['sampling'] = {'seed': config.seed, 'target_selection': 'uniform unique non-source states without replacement',
                           'target_entropy': 'config seed, context index, sha256(context id) prefix',
                           'rollout_entropy': 'config seed, context index, target index, trial index',
                           'rng': 'numpy.random.SeedSequence with PCG64', 'numpy_version': np.__version__,
                           'policy_model_seed': None, 'model_outputs': 'cached realized distributions; model draw is not seeded'}
    summary['median_audit'] = {'rule': 'lower weighted median; integer hit counts; floating-point boundary correction',
                               'changed_contexts': changed, 'corrected_fields': corrected_fields,
                               'reused_trials': len(trials), 'new_api_requests': 0}
    summary['result_digest'] = calibration_result_digest(summary)
    write_json(targets_path, targets)
    write_json(summary_path, summary)
    with (directory / 'calibration.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['context', 'target_index', 'k', 'trials', 'hits', 'p_hat', 'd', 'weight', 'cdf_before',
                  'cdf_at', 'u', 'a_c', 'ratio_d_over_ac', 'normalized_offset']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(examples)
    print(json.dumps({'path': str(directory), **summary['median_audit']}, indent=2))


if __name__ == '__main__':
    main()
