"""Independent prompt, split, D4 and resume checks for the stage-three study.

Historical test responses are never scored or used for any selection here.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_in_context import in_context_messages
from carl.icl.intent_representation_experiment import messages_for
from scripts.overnight_stage03_generate import Transform
from scripts.overnight_stage03_generate import digest
from scripts.overnight_stage03_generate import inverse_coordinate
from scripts.overnight_stage03_generate import map_coordinate
from scripts.overnight_stage03_generate import persist_frozen
from scripts.overnight_stage03_generate import public_record
from scripts.overnight_stage03_generate import replay_transform
from scripts.overnight_stage03_generate import select_contexts
from scripts.overnight_stage03_generate import transform_actions
from scripts.overnight_stage03_generate import transform_rows

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'analysis/overnight_cpu_20261115T061352Z'
DATA = RUN / 'stage_03/data'
ORIGINAL = ROOT / 'analysis/intent_generator_20261114T032232Z'
TRANSFER = ROOT / 'analysis/lora_matrix_20261114T065804Z/eval_transfer'
REPRESENTATIONS = ROOT / 'analysis/intent_representations_20261114T075324Z'


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def fixture_board() -> list[str]:
    cells = np.ones((12, 12), dtype=np.int8)
    cells[0, :] = cells[-1, :] = 0
    cells[:, 0] = cells[:, -1] = 0
    cells[1, 1:5] = 2
    cells[3, 1:5] = 4
    cells[5, 1] = 5
    return to_rows(np.eye(7, dtype=np.float32)[cells])


def reference_pair() -> dict:
    source = fixture_board()
    state = from_rows(source)
    env = environment(state)
    actions = [3, 3]
    target = state
    for action in actions:
        target = np.asarray(env.next_state(target, action))
    return {'pair_id': 'fixture-pair', 'k': len(actions), 'source': source, 'target': to_rows(target),
            'source_hash': state_hash(state), 'target_hash': state_hash(target),
            'layout_id': canonical_layout_id(state), 'demonstration_actions': actions}


def support_record(layout: str, board: list[str]) -> dict:
    return {'layout_id': layout, 'split': 'train', 'validation': {'accepted': True},
            'start_observation': board, 'task_goal': 'Solve the puzzle',
            'intent': {'effects': [{'predicate': 'box_at', 'row': 4, 'col': 4}],
                       'preserve': [], 'instruction': None},
            'later_observation': 'DO_NOT_LEAK_SUPPORT_FUTURE', 'fragment_actions': 'DO_NOT_LEAK_SUPPORT_ACTIONS'}


def test_query_and_support_whitelist_do_not_leak_trajectory_fields():
    pair = reference_pair()
    context = {'layout_id': pair['layout_id'], 'source': pair['source'],
               'later_observation': 'DO_NOT_LEAK_QUERY_FUTURE',
               'fragment_actions': 'DO_NOT_LEAK_QUERY_ACTIONS', 'k': 'DO_NOT_LEAK_QUERY_OFFSET',
               'reference_pair': pair}
    query = public_record(context, Transform.IDENTITY)
    assert set(query) == {'layout_id', 'start_observation', 'task_goal'}
    support = [support_record('different-one', pair['source']), support_record('different-two', pair['source'])]
    messages = in_context_messages(query, support)
    assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'user', 'assistant', 'user']
    assert messages[-1] == prompt_messages(query)[-1]
    assert 'DO_NOT_LEAK_' not in json.dumps(messages)
    support[0]['split'] = 'test'
    with pytest.raises(ValueError, match='training records'):
        in_context_messages(query, support)
    support[0]['split'] = 'train'
    support[0]['layout_id'] = query['layout_id']
    with pytest.raises(ValueError, match='share the query layout'):
        in_context_messages(query, support)


def test_d4_coordinate_action_and_real_transition_roundtrips():
    pair = reference_pair()
    before = pair['source']
    for transform in Transform:
        changed = transform_rows(before, transform)
        assert canonical_layout_id(from_rows(changed)) == pair['layout_id']
        for row in range(12):
            for col in range(12):
                mapped = map_coordinate(row, col, transform)
                assert inverse_coordinate(*mapped, transform) == (row, col)
                assert before[row][col] == changed[mapped[0]][mapped[1]]
        mapped_actions = transform_actions([0, 1, 2, 3], transform)
        assert transform_actions(mapped_actions, transform, inverse=True) == [0, 1, 2, 3]
        assert replay_transform(pair, transform)['actions_replayed'] == 2


def test_split_and_icl_support_layout_conflicts_fail_closed():
    pair = reference_pair()
    common = {**pair, 'context_id': 'one', 'family': '12x12x4', 'source_family': 'source',
              'board_rank': 0, 'context_rank': 0}
    with pytest.raises(ValueError, match='overlaps development and evaluation'):
        select_contexts([{**common, 'split': 'development'}, {**common, 'split': 'evaluation'}],
                        'development', 1, set())
    with pytest.raises(ValueError, match='overlaps an ICL support'):
        select_contexts([{**common, 'split': 'evaluation'}], 'evaluation', 2, {pair['layout_id']})


def test_resume_requires_same_frozen_plan_and_recovers_missing_prompt_file(tmp_path):
    plan = {'cases': [{'case_id': 'one'}]}
    prompts = [{'case_id': 'one', 'messages': [{'role': 'user', 'content': 'board'}]}]
    contract = {'plan_sha256': digest(plan), 'prompts_sha256': digest(prompts)}
    persist_frozen(tmp_path, plan, prompts, contract, resume=False)
    with pytest.raises(ValueError, match='requires --resume'):
        persist_frozen(tmp_path, plan, prompts, contract, resume=False)
    (tmp_path / 'prompts.jsonl').unlink()
    persist_frozen(tmp_path, plan, prompts, contract, resume=True)
    assert read_jsonl(tmp_path / 'prompts.jsonl') == prompts
    with pytest.raises(ValueError, match='disagree with the contract'):
        persist_frozen(tmp_path, {'cases': []}, prompts, contract, resume=True)


@pytest.mark.skipif(not (DATA / 'data_manifest.json').is_file(), reason='campaign artifacts not available')
def test_frozen_historical_prompts_and_splits_without_scoring_outputs():
    contexts = read_jsonl(DATA / 'contexts.jsonl')
    outputs = read_jsonl(DATA / 'outputs.jsonl')
    transfer = {row['record_id']: row for row in read_jsonl(TRANSFER / 'dataset.jsonl')}
    representations = {row['record_id']: row for row in read_jsonl(REPRESENTATIONS / 'data/dataset.jsonl')}
    support_data = {row['record_id']: row for row in read_jsonl(ORIGINAL / 'dataset.jsonl')}
    support_ids = json.loads((ORIGINAL / 'icl_followup/support.json').read_text())['record_ids']
    support = [support_data[identifier] for identifier in support_ids]
    assert len(support) == 2
    assert all(row['split'] == 'train' and row['validation']['accepted'] is True for row in support)
    support_layouts = {row['layout_id'] for row in support}
    assert len(support_layouts) == 2
    test_layouts = {row['layout_id'] for row in contexts}
    assert not (support_layouts & test_layouts)
    stage01 = json.loads((RUN / 'stage_01/data_v2/exposure_register.json').read_text())
    train = json.loads((RUN / 'stage_02/data_train/exposure_register.json').read_text())
    assert not (test_layouts & set(train['train_layout_ids']))
    assert not (support_layouts & set(stage01['campaign_evaluation_layout_ids']))
    by_case = {row['case_id']: row for row in contexts}
    assert len(by_case) == len(contexts) == 168
    for cohort in {'legacy_transfer_40', 'representation_128'}:
        subset = [row for row in contexts if row['cohort_id'] == cohort]
        assert len({row['layout_id'] for row in subset}) == len(subset)
        assert all(canonical_layout_id(from_rows(row['board'])) == row['layout_id'] for row in subset)
    for item in outputs:
        context = by_case[item['case_id']]
        assert item['source_rows'] == context['board'] and item['split'] == context['split'] == 'test'
        if item['prediction_status'] == 'missing':
            assert item['method'] == 'icl_legacy' and item['family'] == '12x12x4'
            continue
        assert item['prediction_status'] == 'complete'
        record_id = item['context_id']
        record = transfer[record_id] if item['cohort_id'] == 'legacy_transfer_40' else representations[record_id]
        if item['cohort_id'] == 'legacy_transfer_40':
            expected = in_context_messages(record, support) if item['method'] == 'icl_legacy' else prompt_messages(record)
        else:
            expected = messages_for(record, support if item['method'] == 'icl_legacy' else [], 'json')
        assert item['prompt']['messages'] == expected
