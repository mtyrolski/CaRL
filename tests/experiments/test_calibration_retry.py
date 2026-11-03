"""Invalid cached policy replies can be preserved and retried without fake trials."""
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.archive.campaign_20261102.resume_calibration_v2 as recovery


def record(text: str, role: str = 'policy') -> dict:
    return {'status': 'completed', 'request_hash': 'a' * 64,
            'request': {'payload': {'metadata': {'component': role}}}, 'output_text': text}


@pytest.mark.parametrize('text', ['{"probabilities":[0,1,0,0}', '{"probabilities":[0,0,0,0]}',
                                  '{"probabilities":[true,0,0,0]}', 'not JSON'])
def test_invalid_policy_json_cannot_be_reused(text: str) -> None:
    assert recovery.invalid_policy_output(record(text))
    assert not recovery.invalid_policy_output(record('{"probabilities":[0,1,0,0]}'))
    assert not recovery.invalid_policy_output(record(text, role='value'))


def test_archive_preserves_invalid_bytes_and_leaves_valid_cache_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / 'calibration'
    directory.mkdir()
    (directory / 'calibration.json').write_text(json.dumps({'status': 'partial', 'error': 'Invalid JSON from model component'}))
    cache = tmp_path / 'cache' / 'openai'
    cache.mkdir(parents=True)
    invalid = record('{"probabilities":[0,1,0,0}')
    digest = hashlib.sha256(json.dumps(invalid['request'], sort_keys=True).encode()).hexdigest()
    invalid['request_hash'] = digest
    active = cache / (digest + '.json')
    original = json.dumps(invalid).encode()
    active.write_bytes(original)
    ledger_path = tmp_path / 'ledger.sqlite3'
    with sqlite3.connect(ledger_path) as db:
        db.execute('create table events(id integer primary key,stage text,tag text,kind text,request_hash text)')
        db.execute('insert into events(stage,tag,kind,request_hash) values (?,?,?,?)',
                   ('calibration', 'calibration_v2', 'logical_request', digest))
    events = []
    ledger = SimpleNamespace(path=ledger_path, event=lambda *args: events.append(args))
    config = SimpleNamespace(path=lambda _field: tmp_path / 'cache')
    monkeypatch.setattr(recovery, 'load_config', lambda _path: config)
    monkeypatch.setattr(recovery, 'calibration_components', lambda *_args: object())
    monkeypatch.setattr(recovery, 'calibration_path', lambda *_args: directory / 'calibration.json')
    monkeypatch.setattr(recovery, 'CampaignLedger', lambda: ledger)
    archived = Path(recovery.archive_invalid('unused'))
    assert archived.read_bytes() == original and not active.exists()
    assert events[0][1] == 'invalid_policy_output_archived'
    valid = record('{"probabilities":[0,1,0,0]}')
    valid['request_hash'] = digest
    active.write_text(json.dumps(valid))
    with pytest.raises(ValueError, match='not an invalid'):
        recovery.archive_invalid('unused')
    assert active.exists()
