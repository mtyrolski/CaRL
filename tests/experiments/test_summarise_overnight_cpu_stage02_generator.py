"""Raw generator failures and duplicate aliases have separate denominators."""
import pytest

from scripts.summarise_overnight_cpu_stage02_generator import proposal_outcomes
from scripts.summarise_overnight_cpu_stage02_generator import summarise


def fixtures():
    context = {'context_id': 'c', 'layout_id': 'layout', 'family': '12x12x4', 'source_family': 'source',
               'split': 'evaluation', 'board_rank': 0, 'generator_shard_missing': False,
               'raw_proposals': [{'rank': 0, 'status': 'valid_unique', 'candidate_id': 'goal',
                                  'target_hash': 'target', 'validation_error': None},
                                 {'rank': 1, 'status': 'duplicate', 'candidate_id': 'goal',
                                  'target_hash': 'target', 'validation_error': None}]}
    plan = {'planned_attempts': [{'controller': model, 'candidate_id': 'goal'} for model in ('cllp8', 'cllp32')]}
    attempts = [{'controller': model, 'candidate_id': 'goal', 'status': 'complete', 'exact_hit': True,
                 'first_hit': 2, 'attempted_actions': 2, 'actions': [0, 1], 'state_hashes': ['a', 'b', 'target'],
                 'stop_reason': 'exact_target'} for model in ('cllp8', 'cllp32')]
    return context, plan, attempts


def test_duplicate_raw_outputs_reuse_one_attempt_per_controller():
    context, plan, attempts = fixtures()
    rows = proposal_outcomes([context], plan, attempts)
    selected = [row for row in rows if row['controller'] == 'cllp8' and row['L'] == 8]
    assert len(selected) == 2 and all(row['p'] == 1 for row in selected)
    summary = next(row for row in summarise(rows)
                   if row['source_family'] == 'all' and row['controller'] == 'cllp8' and row['L'] == 8)
    assert summary['successful_outputs'] == 2 and summary['duplicate_outputs'] == 1
    assert summary['bootstrap_layouts'] == 1 and summary['ci_low'] is None


def test_invalid_and_missing_are_not_controller_failures():
    context, plan, attempts = fixtures()
    context['raw_proposals'][1] = {'rank': 1, 'status': 'invalid', 'candidate_id': None,
                                   'target_hash': None, 'validation_error': 'box count'}
    rows = proposal_outcomes([context], plan, attempts[:1])
    invalid = next(row for row in rows if row['rank'] == 1 and row['controller'] == 'cllp8' and row['L'] == 8)
    assert invalid['controller_status'] == 'not_applicable' and invalid['p'] is None
    missing = next(row for row in rows if row['rank'] == 0 and row['controller'] == 'cllp32' and row['L'] == 8)
    assert missing['controller_status'] == 'missing' and missing['p'] is None
    summary = next(row for row in summarise(rows)
                   if row['source_family'] == 'all' and row['controller'] == 'cllp32' and row['L'] == 8)
    assert summary['valid_missing_controller'] == 1 and summary['complete_contexts'] == 0


def test_valid_proposal_missing_from_plan_fails_closed():
    context, plan, attempts = fixtures()
    plan['planned_attempts'].pop()
    with pytest.raises(ValueError, match='omitted'):
        proposal_outcomes([context], plan, attempts[:1])
