"""Checks for balanced data selection and generator input isolation."""

from collections import Counter

from carl.icl.intent_generator_experiment import prompt_messages
from scripts.prepare_overnight_cpu_stage04_data import CATEGORIES
from scripts.prepare_overnight_cpu_stage04_data import allocate
from scripts.prepare_overnight_cpu_stage04_data import category_quotas
from scripts.prepare_overnight_cpu_stage04_data import describe_marginals
from scripts.prepare_overnight_cpu_stage04_data import matching_allocations


def pools(boards: int, per_category: int) -> dict:
    return {f'board_{i:03}': {kind: [(j, 1, kind, {'index': j}) for j in range(per_category)]
                               for kind in CATEGORIES} for i in range(boards)}


def counts(allocation: dict) -> Counter:
    return Counter(item[2] for items in allocation.values() for item in items)


def test_matched_allocation_uses_all_boards_and_identical_categories():
    small, wide = pools(9, 64), pools(64, 12)
    total, quotas, narrow_result, wide_result = matching_allocations(small, wide)
    assert total == 256
    assert counts(narrow_result) == counts(wide_result) == quotas
    assert all(1 <= len(items) <= 32 for items in narrow_result.values())
    assert all(len(items) == 4 for items in wide_result.values())
    assert quotas[CATEGORIES[-1]] / total <= .10


def test_infeasible_category_allocation_returns_none():
    pool = {'board': {kind: [] for kind in CATEGORIES}}
    assert allocate(pool, category_quotas(10), cap=10) is None


def test_residual_marginals_distinguish_goal_thresholds_and_coordinates():
    row = {'intent': {'effects': [{'predicate': 'box_at', 'row': 3, 'col': 4},
                                   {'predicate': 'boxes_on_goals_at_least', 'count': 2}],
                      'preserve': [], 'instruction': None}}
    result = describe_marginals([row])
    assert result['conjunction_sizes'] == {'2': 1}
    assert result['goal_threshold_counts'] == {'2': 1}
    assert result['coordinate_counts'] == {'3,4': 1}


def test_generator_prompt_whitelists_board_and_task_only():
    record = {'start_observation': ['###', '#@#', '###'], 'task_goal': 'Put every box on a goal.',
              'later_observation': ['NEVER-READ-FUTURE'], 'fragment_actions': ['NEVER-READ-ACTION'],
              'i': 'NEVER-READ-INDEX', 'k': 'NEVER-READ-OFFSET', 'trajectory_id': 'NEVER-READ-ID'}
    messages = prompt_messages(record)
    assert 'NEVER-READ' not in str(messages)
    assert '#@#' in messages[-1]['content']
