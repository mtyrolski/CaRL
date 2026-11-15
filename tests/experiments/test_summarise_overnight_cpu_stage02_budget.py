"""Budget sensitivity uses matched targets and independent board clusters."""
import copy

import pytest

from scripts.summarise_overnight_cpu_stage02_budget import compare
from scripts.summarise_overnight_cpu_stage02_budget import summarise


def profiles():
    meta = {'context_id': 'c', 'layout_id': 'board', 'family': '12x12x4', 'source_family': 'source',
            'split': 'evaluation', 'controller': 'cllp8', 'pool_variant': 'demo4_bfs4',
            'pool': 'fixed', 'status': 'complete'}
    short = [{'candidate_id': 'a', 'p': 1, 'd': 4, 'u': .5},
             {'candidate_id': 'b', 'p': 0, 'd': None, 'u': None}]
    long = [{'candidate_id': 'a', 'p': 1, 'd': 4, 'u': .25},
            {'candidate_id': 'b', 'p': 1, 'd': 12, 'u': .75}]
    return [{**meta, 'L': 8, 'candidates': short}, {**meta, 'L': 16, 'candidates': long},
            {**meta, 'L': 32, 'candidates': copy.deepcopy(long)}]


def test_same_target_scale_changes_when_longer_limit_adds_success():
    contexts, targets = compare(profiles())
    row = next(item for item in contexts if item['short_L'] == 8 and item['long_L'] == 32)
    assert row['shared_success_targets'] == 1 and row['new_success_targets'] == 1
    assert row['mean_abs_delta_u'] == .25
    target = next(item for item in targets if item['short_L'] == 8 and item['long_L'] == 32)
    assert target['candidate_id'] == 'a' and target['delta_u'] == -.25
    result = next(item for item in summarise(contexts) if item['source_family'] == 'all'
                  and item['short_L'] == 8 and item['long_L'] == 32)
    assert result['complete_contexts'] == 1 and result['delta_layouts'] == 1
    assert result['ci_low'] is None and result['new_success_fraction'] == 1


def test_missing_profile_is_not_turned_into_a_failure():
    data = profiles()
    data[0]['status'] = 'incomplete'
    contexts, _ = compare(data)
    row = next(item for item in contexts if item['short_L'] == 8 and item['long_L'] == 32)
    assert row['status'] == 'unavailable' and row['new_success_targets'] is None


def test_inconsistent_success_prefix_is_rejected():
    data = profiles()
    data[1]['candidates'][0]['p'] = 0
    with pytest.raises(ValueError, match='shorter deterministic prefix'):
        compare(data)
