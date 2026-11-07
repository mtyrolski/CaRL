"""Exercise Figure 1 accounting against isolated temporary databases only."""
import hashlib
import json
import multiprocessing as mp
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from carl.icl.campaign_budget import CampaignLimitExceeded
from scripts.figure1_pilot.budget import Figure1Ledger


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr('carl.icl.campaign_budget.dotenv_values', lambda _path: {})
    monkeypatch.delenv('CARL_CAMPAIGN_MAX_USD', raising=False)
    price = {'input_usd_per_million': '1', 'cached_input_usd_per_million': '.1',
             'cache_write_30m_usd_per_million': '1', 'cache_write_12h_usd_per_million': '1',
             'output_usd_per_million': '1', 'max_input_tokens': 100, 'max_output_tokens': 128,
             'long_context_threshold': 90, 'regional_multiplier': '1'}
    body = {'campaign_id': 'figure1_test_only', 'authorization_reference': 'test-only',
            'ledger_path': str(tmp_path / 'guard.sqlite3'), 'limit_usd': '1',
            'stage_limits_usd': {'pilot': '1'},
            'model_limits_usd': {'gpt-6-luna': '.0003', 'gpt-6.1-terra': '.0003'},
            'models': {'gpt-6-luna': price, 'gpt-6.1-terra': price.copy()}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(body))
    return path, body


def payload(model='gpt-6.1-terra'):
    return {'model': model, 'input': 'test', 'max_output_tokens': 32}


def test_quote_reserve_and_settle_exact_worst_case(config):
    path, _ = config
    ledger = Figure1Ledger(path)
    request = payload()
    assert ledger.quote_reservation('responses', request) == 248000
    identifier = ledger.reserve('pilot', 'responses', request, 'h')
    with ledger._transaction() as db:
        assert db.execute('SELECT reserved_nanos FROM attempts WHERE id=?', (identifier,)).fetchone()[0] == 248000
    assert ledger.settle(identifier, {'input_tokens': 100, 'output_tokens': 32}, request)
    report = ledger.summary()
    assert report['by_model_stage_status'][0]['accounted_cost_usd'] == .000248
    assert 'not necessarily the provider invoice' in report['cost_note']
    assert report['settled_usage_by_model']['gpt-6.1-terra']['input_tokens'] == 100
    assert report['settled_usage_by_model']['gpt-6.1-terra']['provider_cached_input_tokens'] is None
    assert report['settled_usage_by_model']['gpt-6.1-terra']['provider_cache_write_tokens'] is None


def test_settlement_must_match_original_request(config):
    path, _ = config
    ledger = Figure1Ledger(path)
    original = payload()
    identifier = ledger.reserve('pilot', 'responses', original, 'h')
    with pytest.raises(ValueError, match='original reservation'):
        ledger.settle(identifier, {'input_tokens': 1, 'output_tokens': 1}, payload('gpt-6-luna'))
    changed = {**original, 'max_output_tokens': 1}
    with pytest.raises(ValueError, match='original reservation'):
        ledger.settle(identifier, {'input_tokens': 1, 'output_tokens': 1}, changed)
    with ledger._transaction() as db:
        assert db.execute('SELECT status, charged_nanos FROM attempts WHERE id=?', (identifier,)).fetchone() == ('reserved', None)


def test_provider_cache_usage_is_separate_from_local_cache_events(config):
    path, _ = config
    ledger = Figure1Ledger(path)
    identifier = ledger.reserve('pilot', 'responses', payload(), 'h')
    usage = {'input_tokens': 30, 'output_tokens': 10,
             'input_tokens_details': {'cached_tokens': 10, 'cache_write_tokens': 5}}
    assert ledger.settle(identifier, usage, payload())
    ledger.event('pilot', 'live_cache_hit', 'responses', 'h')
    report = ledger.summary()
    totals = report['settled_usage_by_model']['gpt-6.1-terra']
    assert totals['provider_cached_input_tokens'] == 10
    assert totals['provider_cache_write_tokens'] == 5
    assert report['events'][0]['kind'] == 'live_cache_hit'


@pytest.mark.parametrize('usage', [
    {}, {'input_tokens': None, 'output_tokens': 1}, {'input_tokens': True, 'output_tokens': 1},
    {'input_tokens': -1, 'output_tokens': 1}, {'input_tokens': 1, 'output_tokens': 33},
    {'input_tokens': 1, 'output_tokens': 1, 'input_tokens_details': {'cached_tokens': True}},
    {'input_tokens': 1, 'output_tokens': 1, 'input_tokens_details': {'cached_tokens': 2, 'cache_write_tokens': 0}},
])
def test_invalid_usage_never_releases_reservation(config, usage):
    path, _ = config
    ledger = Figure1Ledger(path)
    identifier = ledger.reserve('pilot', 'responses', payload(), 'h')
    assert not ledger.settle(identifier, usage, payload())
    ledger.mark_unknown(identifier)
    with ledger._transaction() as db:
        assert db.execute('SELECT status, charged_nanos FROM attempts WHERE id=?', (identifier,)).fetchone() == ('unknown', None)


@pytest.mark.parametrize(('field', 'value'), [('cache_write_12h_usd_per_million', '.5'),
                                              ('regional_multiplier', '.9'),
                                              ('max_input_tokens', 0),
                                              ('max_output_tokens', True),
                                              ('output_usd_per_million', '0')])
def test_underpricing_or_invalid_caps_rejected_before_ledger_creation(config, field, value):
    path, body = config
    body['models']['gpt-6.1-terra'][field] = value
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        Figure1Ledger(path)
    assert not (path.parent / 'guard.sqlite3').exists()


def test_existing_unrelated_database_is_never_modified(config, tmp_path):
    path, body = config
    existing = tmp_path / 'unrelated.sqlite3'
    with sqlite3.connect(existing) as db:
        db.execute('CREATE TABLE important (value TEXT)')
        db.execute("INSERT INTO important VALUES ('preserve')")
    original_hash = hashlib.sha256(existing.read_bytes()).hexdigest()
    body['ledger_path'] = str(existing)
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='not a matching'):
        Figure1Ledger(path)
    assert hashlib.sha256(existing.read_bytes()).hexdigest() == original_hash


def test_atomic_cross_instance_model_caps(config):
    path, _ = config
    first, second = Figure1Ledger(path), Figure1Ledger(path)
    def reserve(index):
        try:
            (first if index % 2 == 0 else second).reserve('pilot', 'responses', payload(), str(index))
            return True
        except CampaignLimitExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as workers:
        accepted = list(workers.map(reserve, range(8)))
    assert sum(accepted) == 1
    with first._transaction() as db:
        assert db.execute('SELECT COUNT(*), SUM(reserved_nanos) FROM attempts').fetchone() == (1, 248000)


def test_atomic_caps_across_separate_processes(config):
    path, _ = config
    Figure1Ledger(path)
    context = mp.get_context('fork')
    barrier = context.Barrier(2)
    output = context.Queue()

    def reserve_in_process():
        ledger = Figure1Ledger(path)
        barrier.wait(timeout=10)
        try:
            ledger.reserve('pilot', 'responses', payload(), 'same-request')
        except CampaignLimitExceeded:
            output.put('rejected')
        else:
            output.put('reserved')

    workers = [context.Process(target=reserve_in_process) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    assert sorted(output.get(timeout=2) for _ in workers) == ['rejected', 'reserved']
    ledger = Figure1Ledger(path)
    with ledger._transaction() as db:
        assert db.execute('SELECT COUNT(*),SUM(reserved_nanos) FROM attempts').fetchone() == (1, 248000)
