"""Hand-calculated checks independent of runner rollouts and output caches."""
from types import SimpleNamespace

import pytest

from scripts.reachability_runner.engine import _metric_row, calibration_metrics, paired_metrics


def row(layout, score, *, group='main', arm='luna', k=1, pair='p', rep=0, time=2, context='ctx'):
    job = {'layout': layout, 'group': group, 'arm': arm, 'k': k, 'pair': pair,
           'repetition': rep, 'context': context}
    if score == 'missing':
        return {'job': job, 'status': 'missing'}
    if score == 'invalid':
        return {'job': job, 'status': 'invalid_model_response'}
    return {'job': job, 'status': 'complete', 'exact_hit': bool(score),
            'stop_reason': 'exact_hit' if score else 'miss_cap', 'first_hit': time if score else None}


def test_partial_cell_is_not_a_formal_point_estimate():
    rows = [row('a', 1), row('a', 0, rep=1), row('a', 'invalid', rep=2), row('a', 'missing', rep=3)]
    result = _metric_row(rows, arm='luna', stage='main', k=1)
    assert result['status'] == 'PARTIAL' and result['operational_p'] is None
    assert result['resolved_p_descriptive'] == pytest.approx(1/3)
    assert result['coverage'] == .75 and result['mean_first_hit_given_hit'] == 2
    assert (result['lower_bound'], result['upper_bound']) == (.25, .5)
    assert result['invalid_model_response'] == result['unresolved'] == 1


def test_equal_layout_weights_and_zero_hit():
    rows = [row('a', 1), *[row('b', 0, rep=i) for i in range(3)]]
    result = _metric_row(rows, arm='luna', stage='main', k=1)
    assert result['operational_p'] == .5  # (1/1 + 0/3)/2, not 1/4
    assert result['planned_layouts'] == 2 and result['planned'] == 4
    zero = _metric_row([row('a', 0)], arm='luna', stage='main', k=1)
    assert zero['operational_p'] == 0 and zero['mean_first_hit_given_hit'] is None
    assert zero['zero_hit_warning']


def test_unequal_repetitions_and_tied_midpoints():
    # p_A=1, p_B=1/2, p_C=1/4, p_Z=0, times A=B=2, C=4.
    rows = [row('a', 1, group='calibration_A', pair='A', time=2),
            row('a', 1, group='calibration_A', pair='B', time=2),
            row('a', 0, group='calibration_A', pair='B', rep=1),
            row('a', 1, group='calibration_A', pair='C', time=4),
            *[row('a', 0, group='calibration_A', pair='C', rep=i) for i in range(1, 4)],
            row('a', 0, group='calibration_A', pair='Z')]
    targets = {x['pair']: x for x in calibration_metrics(rows)}
    assert [targets[p]['weight'] for p in 'ABC'] == pytest.approx([4/7, 2/7, 1/7])
    assert [targets[p]['u'] for p in 'ABC'] == pytest.approx([3/7, 3/7, 13/14])
    assert targets['A']['cdf_before'] == 0 and targets['B']['cdf_at'] == pytest.approx(6/7)
    assert targets['Z']['p'] == 0 and targets['Z']['d'] is None and targets['Z']['u'] is None
    assert targets['Z']['exclusion_reason'] == 'zero_exact_hits'
    rows[-1] = row('a', 'missing', group='calibration_A', pair='Z')
    targets = calibration_metrics(rows)
    assert all(t['context_status'] == 'PARTIAL' and t['u'] is None for t in targets)


def test_paired_bootstrap_equal_layouts_and_shared_across_k():
    rows = []
    for k in [1, 2]:
        for layout, repetitions, first, second in [('a', 1, 0, 1), ('b', 2, 1, 0)]:
            for rep in range(repetitions):
                rows.extend([row(layout, first, arm='luna', k=k, pair=f'p{k}', rep=rep),
                             row(layout, second, arm='terra_6_1', k=k, pair=f'p{k}', rep=rep)])
    result = paired_metrics(rows, SimpleNamespace(core_offsets=[1, 2]), 'contract')
    assert len(result) == 2
    assert all(x['delta_terra_minus_luna'] == 0 for x in result)  # equal layouts, not -1/3
    assert all(x['planned_layouts'] == 2 and x['planned_pairs'] == 3 for x in result)
    assert result[0]['bootstrap_95_low'] == result[1]['bootstrap_95_low']
    assert result[0]['bootstrap_95_high'] == result[1]['bootstrap_95_high']
    assert all(x['bootstrap_repetitions'] == 10000 for x in result)
    rows[-1]['status'] = 'missing'
    result = paired_metrics(rows, SimpleNamespace(core_offsets=[1, 2]), 'contract')
    assert all(x['bootstrap_95_low'] is None for x in result)  # partial stage forbids all CIs
    assert result[1]['status'] == 'PARTIAL' and result[1]['delta_terra_minus_luna'] is None
    assert result[1]['full_cohort_lower_bound'] <= 0 <= result[1]['full_cohort_upper_bound']
