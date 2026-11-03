"""Small offline checks for the frozen 32x32 controller diagnostic."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from carl.icl.api import BudgetExceeded
from carl.icl.campaign_budget import CampaignLedger
from carl.icl.config import Limits
from scripts.campaign_bfs32_cllp import DESIGN, ROOT, DiagnosticAPI, selected_cases


def test_cases_exclude_identity_and_keep_bfs_out_of_policy_handoff() -> None:
    design = json.loads(DESIGN.read_text())
    cases, excluded = selected_cases(design)
    assert len(cases) == 4
    assert len(excluded) == 1 and excluded[0]['reason'] == 'source_equals_target'
    handoff = json.loads((ROOT / design['policy_inputs']).read_text())
    assert all(set(row) == {'board_id', 'offset_id', 'source', 'target'} for row in handoff['cases'])
    assert all(row['source_hash'] != row['target_hash'] for row in cases)


def test_small_cost_cap_stops_before_provider_call(tmp_path: Path) -> None:
    ledger_path = tmp_path / 'cost.sqlite3'
    with sqlite3.connect(ledger_path) as db:
        db.execute('CREATE TABLE attempts (stage TEXT, tag TEXT, charged_nanos INTEGER, reserved_nanos INTEGER)')
        db.execute('INSERT INTO attempts VALUES (?,?,?,?)', ('policy_validation', 'bfs32_cllp_v1:test', 180000000, 0))
    api = DiagnosticAPI(tmp_path, 'mock', Limits(), cap_usd=.75, tag_prefix='bfs32_cllp_v1:')
    api.provider = 'openai'
    api.stage = 'policy_validation'
    api.ledger = cast(CampaignLedger, SimpleNamespace(path=ledger_path, config={'models': {'gpt-6-luna': {
        'cache_write_12h_usd_per_million': '.25', 'max_input_tokens': 1050000,
        'output_usd_per_million': '.50', 'regional_multiplier': '1.10'}}}))
    with pytest.raises(BudgetExceeded, match='diagnostic cost cap'):
        api.call('responses', {'model': 'gpt-6-luna', 'max_output_tokens': 2048})


def test_mock_prepare_execute_and_resume(tmp_path: Path) -> None:
    command = [sys.executable, '-m', 'scripts.campaign_bfs32_cllp', '--provider', 'mock', '--output', str(tmp_path)]
    subprocess.run([*command, '--prepare'], cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run([*command, '--execute'], cwd=ROOT, check=True, capture_output=True, text=True)
    summary_path = tmp_path / 'mock' / 'summary.json'
    result = json.loads(summary_path.read_text())
    assert result['status'] == 'completed' and result['expected'] == result['completed'] == 4
    assert result['missing'] == result['interrupted'] == 0
    rows = list((tmp_path / 'mock').glob('*_offset*.json'))
    assert len(rows) == 4 and all(json.loads(path.read_text())['trace_replayed'] for path in rows)
    cached = list((tmp_path / 'mock' / 'mock_cache' / 'mock').glob('*.json'))
    assert cached
    for path in cached:
        request = json.loads(path.read_text())['request']['payload']
        assert request['metadata']['component'] == 'policy'
        for message in request['input']:
            if message['role'] == 'user':
                query = json.loads(message['content'][0]['text'])
                assert set(query) == {'state', 'subgoal', 'state_coordinates', 'subgoal_coordinates'}
    timestamps = {path.name: path.stat().st_mtime_ns for path in rows}
    subprocess.run([*command, '--execute'], cwd=ROOT, check=True, capture_output=True, text=True)
    assert timestamps == {path.name: path.stat().st_mtime_ns for path in rows}
