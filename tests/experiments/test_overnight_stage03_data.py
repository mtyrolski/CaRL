"""Checks for frozen historical generator cohort identity and provenance."""

import json

import pytest

from scripts.overnight_stage03_data import COHORT_METHODS
from scripts.overnight_stage03_data import Cohort
from scripts.overnight_stage03_data import Method
from scripts.overnight_stage03_data import context_row
from scripts.overnight_stage03_data import read_selected_data
from scripts.overnight_stage03_data import sha


def record(identifier: str, layout: str) -> dict:
    return {'record_id': identifier, 'layout_id': layout, 'split': 'test',
            'evaluation_family': 'real_12x12x4', 'start_observation': ['############'] +
            ['#----------#'] * 2 + ['#$$$$------#'] + ['#----------#'] * 7 + ['############'],
            'source_path': 'fixture', 'source_sha256': 'sha', 'task_goal': 'Solve puzzle'}


def test_context_is_stable_and_predeclares_all_method_arms():
    item = context_row(Cohort.LEGACY, record('record_a', 'layout_a'))
    assert item['case_id'] == 'legacy_transfer_40:record_a:identity'
    assert item['context_id'] == 'record_a'
    assert item['expected_methods'] == [Method.BASE, Method.ICL, Method.ADAPTER]
    assert item['source_rows'] == item['board']
    assert COHORT_METHODS[Cohort.REPRESENTATION] == (Method.BASE, Method.ICL)


def test_frozen_ids_and_unique_layouts_are_checked(tmp_path):
    data = tmp_path / 'data.jsonl'
    ids = tmp_path / 'ids.json'
    rows = [record('a', 'one'), record('b', 'two')]
    data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    ids.write_text(json.dumps({'record_ids': ['b', 'a'], 'dataset_sha256': sha(data)}))
    assert [row['record_id'] for row in read_selected_data(data, ids, 2)] == ['b', 'a']
    rows[1]['layout_id'] = rows[0]['layout_id']
    data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    ids.write_text(json.dumps({'record_ids': ['b', 'a'], 'dataset_sha256': sha(data)}))
    with pytest.raises(ValueError, match='one frozen test context'):
        read_selected_data(data, ids, 2)
    ids.write_text(json.dumps({'record_ids': ['b', 'a'], 'dataset_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='another dataset'):
        read_selected_data(data, ids, 2)
