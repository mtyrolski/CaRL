"""Fake-provider tests for dual-budget Figure 1 transport. No requests leave the process."""
import hashlib
import json
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from carl.icl.reachability_data import canonical_bytes
from scripts.figure1_pilot.budget import Figure1Ledger
from scripts.figure1_pilot.transport import LiveTransportError, call_responses
from scripts.reachability_runner.ledger import OfflineLedger


class FakeUsage:
    def model_dump(self, **_kwargs):
        return {'input_tokens': 3, 'output_tokens': 2, 'input_tokens_details': {'cached_tokens': 1}}


class FakeResponse:
    status = 'completed'
    id = 'response-fixture'
    output_text = '{"probabilities":[0,0,0,1]}'
    output = []
    usage = FakeUsage()
    def __init__(self, model='gpt-6.1-terra'):
        self.model = model
    def model_dump(self, **_kwargs):
        return {'id': self.id, 'model': self.model, 'status': self.status,
                'output_text': self.output_text, 'usage': self.usage.model_dump()}


class FakeClient:
    max_retries = 0
    def __init__(self, response=None, error=None):
        self.responses = self
        self.response = response or FakeResponse()
        self.error = error
        self.calls = 0
        self.lock = threading.Lock()
    def create(self, **kwargs):
        assert kwargs['timeout'] == 10
        with self.lock:
            self.calls += 1
        if self.error:
            raise self.error
        return self.response


def _process_request(context, counter, queue):
    class SharedClient(FakeClient):
        def create(self, **kwargs):
            with counter.get_lock():
                counter.value += 1
            return super().create(**kwargs)
    context['client'] = SharedClient()
    try:
        queue.put(('ok', call_responses(**context)[1]['cache_hit']))
    except Exception as exc:
        queue.put(('error', type(exc).__name__))


def _hard_crash_request(context):
    class CrashClient(FakeClient):
        def create(self, **_kwargs):
            os._exit(7)
    context['client'] = CrashClient()
    call_responses(**context)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'socket', lambda *_a, **_k: pytest.fail('network access forbidden'))
    monkeypatch.setattr('carl.icl.campaign_budget.dotenv_values', lambda _path: {})
    monkeypatch.delenv('CARL_CAMPAIGN_MAX_USD', raising=False)


@pytest.fixture
def context(tmp_path):
    price = {'input_usd_per_million': '1', 'cached_input_usd_per_million': '.1',
             'cache_write_30m_usd_per_million': '1', 'cache_write_12h_usd_per_million': '1',
             'output_usd_per_million': '1', 'max_input_tokens': 100, 'max_output_tokens': 128,
             'long_context_threshold': 90, 'regional_multiplier': '1'}
    config = {'campaign_id': 'figure1_test_transport', 'authorization_reference': 'test-only',
              'ledger_path': str(tmp_path / 'campaign.sqlite3'), 'limit_usd': '1',
              'stage_limits_usd': {'pilot': '1'},
              'model_limits_usd': {'gpt-6-luna': '1', 'gpt-6.1-terra': '1'},
              'models': {'gpt-6-luna': price, 'gpt-6.1-terra': price}}
    config_path = tmp_path / 'campaign.json'
    config_path.write_text(json.dumps(config))
    campaign = Figure1Ledger(config_path)
    limits = {'logical_calls': 100, 'physical_calls': 100, 'total_usd': '1',
              'arm_usd': {'terra_6_1': '1'}, 'stage_usd': {'pilot': '1'}}
    study = OfflineLedger(tmp_path / 'study.sqlite3', limits, create=True)
    request = {'profile_sha256': 'profile', 'endpoint': 'responses', 'slot': 0,
               'payload': {'model': 'gpt-6.1-terra', 'input': 'example', 'max_output_tokens': 32, 'stream': False}}
    digest = hashlib.sha256(canonical_bytes(request)).hexdigest()
    return {'request': request, 'cache_file': tmp_path / 'cache' / (digest + '.json'),
            'study_ledger': study, 'campaign_ledger': campaign,
            'job': {'key': 'trial', 'group': 'pilot', 'arm': 'terra_6_1', 'model': 'gpt-6.1-terra'},
            'attempt': 1, 'client': FakeClient(), 'allowed_response_models': ('gpt-6.1-terra',),
            'timeout_seconds': 10}


def test_completed_request_is_frozen_and_resampled_cache_hit(context):
    response, trace = call_responses(**context)
    again, cached_trace = call_responses(**context)
    assert response == again
    assert response['requested_model'] == response['model'] == 'gpt-6.1-terra'
    assert response['response']['id'] == 'response-fixture' and response['usage']['input_tokens'] == 3
    assert not trace['cache_hit'] and cached_trace['cache_hit'] and context['client'].calls == 1
    assert context['study_ledger'].summary()['local_cache_hits'] == 1
    with context['campaign_ledger']._transaction() as db:
        status, cost = db.execute('SELECT status,charged_nanos FROM attempts').fetchone()
    assert status == 'settled' and cost == 5000


def test_modified_cached_policy_is_rejected_before_use(context):
    call_responses(**context)
    cached = json.loads(context['cache_file'].read_text())
    cached['output_text'] = '{"probabilities":[1,0,0,0]}'
    context['cache_file'].write_text(json.dumps(cached))
    with pytest.raises(LiveTransportError, match='cached_campaign_mismatch_requires_reconciliation'):
        call_responses(**context)
    assert context['client'].calls == 1


def test_concurrent_get_or_create_sends_once(context):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: call_responses(**context), range(8)))
    assert context['client'].calls == 1 and sum(not t['cache_hit'] for _, t in results) == 1
    assert context['study_ledger'].summary()['local_cache_hits'] == 7


def test_two_processes_share_single_physical_request(context):
    mp = multiprocessing.get_context('fork')
    counter = mp.Value('i', 0)
    queue = mp.Queue()
    processes = [mp.Process(target=_process_request, args=(context, counter, queue)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=1) for _ in processes]
    assert all(result[0] == 'ok' for result in outcomes)
    assert sorted(result[1] for result in outcomes) == [False, True]
    assert counter.value == 1


def test_timeout_keeps_both_reservations_and_never_sends_again(context):
    context['client'].error = TimeoutError('sensitive exception message')
    with pytest.raises(LiveTransportError, match='timeout_billing_unknown'):
        call_responses(**context)
    assert context['study_ledger'].summary()['unresolved_reservations'] == 1
    with context['campaign_ledger']._transaction() as db:
        status, charged = db.execute('SELECT status,charged_nanos FROM attempts').fetchone()
    assert status == 'unknown' and charged is None
    assert 'sensitive exception message' not in context['cache_file'].read_text()
    with pytest.raises(LiveTransportError, match='cached_error_requires_reconciliation'):
        call_responses(**context)
    assert context['client'].calls == 1
    assert context['study_ledger'].summary()['logical_calls'] == 2


def test_openai_sdk_timeout_is_a_timeout_not_a_generic_error(context):
    import httpx
    import openai

    context['client'].error = openai.APITimeoutError(request=httpx.Request('POST', 'https://api.openai.com/v1/responses'))
    with pytest.raises(LiveTransportError, match='timeout_billing_unknown'):
        call_responses(**context)
    assert context['study_ledger'].summary()['unresolved_reservations'] == 1


def test_unexpected_model_does_not_become_policy_or_release_reservations(context):
    context['client'].response = FakeResponse('gpt-6-luna')
    with pytest.raises(LiveTransportError, match='unexpected_resolved_model'):
        call_responses(**context)
    with context['campaign_ledger']._transaction() as db:
        status, charged = db.execute('SELECT status,charged_nanos FROM attempts').fetchone()
    assert status == 'unknown' and charged is None
    assert context['study_ledger'].summary()['unresolved_reservations'] == 1
    cached = json.loads(context['cache_file'].read_text())
    assert cached['reason'] == 'unexpected_resolved_model' and cached['model'] == 'gpt-6-luna'
    with pytest.raises(LiveTransportError, match='cached_error_requires_reconciliation'):
        call_responses(**context)
    assert context['client'].calls == 1


@pytest.mark.parametrize('returned_model', [None, 'gpt-6-luna'])
def test_incomplete_response_without_allowed_model_retains_unknown_cost(context, returned_model):
    response = FakeResponse(returned_model)
    response.status = 'incomplete'
    context['client'].response = response
    with pytest.raises(LiveTransportError, match='provider_incomplete_or_refused'):
        call_responses(**context)
    assert context['study_ledger'].summary()['unresolved_reservations'] == 1
    with context['campaign_ledger']._transaction() as db:
        assert db.execute('SELECT status,charged_nanos FROM attempts').fetchone() == ('unknown', None)


def test_completed_cache_from_another_campaign_is_not_consumed(context, tmp_path):
    call_responses(**context)
    original = context['campaign_ledger']
    config = dict(original.config)
    config.update(campaign_id='figure1_other_test', ledger_path=str(tmp_path / 'other.sqlite3'))
    other_path = tmp_path / 'other.json'
    other_path.write_text(json.dumps(config))
    context['campaign_ledger'] = Figure1Ledger(other_path)
    with pytest.raises(LiveTransportError, match='cached_campaign_mismatch_requires_reconciliation'):
        call_responses(**context)
    assert context['client'].calls == 1
    with context['study_ledger'].transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM calls WHERE status='cache_error'").fetchone()[0] == 1


def test_completed_cache_with_changed_usage_is_not_consumed(context):
    call_responses(**context)
    cached = json.loads(context['cache_file'].read_text())
    cached['usage']['input_tokens'] = 4
    context['cache_file'].write_text(json.dumps(cached))
    with pytest.raises(LiveTransportError, match='cached_campaign_mismatch_requires_reconciliation'):
        call_responses(**context)
    assert context['client'].calls == 1


def test_crash_after_reservations_blocks_next_attempt(context):
    context['client'].error = SystemExit('simulated crash')
    with pytest.raises(SystemExit):
        call_responses(**context)
    assert json.loads(context['cache_file'].read_text())['status'] == 'pending'
    assert context['study_ledger'].summary()['unresolved_reservations'] == 1
    with context['campaign_ledger']._transaction() as db:
        assert db.execute('SELECT status FROM attempts').fetchone()[0] == 'reserved'
    context['client'].error = None
    with pytest.raises(LiveTransportError, match='cached_error_requires_reconciliation'):
        call_responses(**context)
    assert context['client'].calls == 1


def test_hard_process_exit_preserves_pending_and_prevents_resend(context):
    process = multiprocessing.get_context('fork').Process(target=_hard_crash_request, args=(context,))
    process.start()
    process.join(timeout=15)
    assert process.exitcode == 7
    assert json.loads(context['cache_file'].read_text())['status'] == 'pending'
    assert context['study_ledger'].summary()['unresolved_reservations'] == 1
    with context['campaign_ledger']._transaction() as db:
        assert db.execute('SELECT status FROM attempts').fetchone()[0] == 'reserved'
    with pytest.raises(LiveTransportError, match='cached_error_requires_reconciliation'):
        call_responses(**context)
    assert context['client'].calls == 0


def test_disabled_sdk_retry_is_mandatory(context):
    context['client'].max_retries = 2
    with pytest.raises(ValueError, match='retries'):
        call_responses(**context)
    assert context['client'].calls == 0


def test_pending_and_both_reservations_are_durable_before_send(context):
    original = context['client'].create
    def checking_create(**kwargs):
        pending = json.loads(context['cache_file'].read_text())
        assert pending['status'] == 'pending'
        with context['campaign_ledger']._transaction() as db:
            assert db.execute('SELECT status FROM attempts WHERE id=?', (pending['campaign_reservation_id'],)).fetchone()[0] == 'reserved'
        with context['study_ledger'].transaction() as db:
            assert db.execute('SELECT status FROM calls WHERE id=?', (pending['study_ledger_id'],)).fetchone()[0] == 'reserved'
        return original(**kwargs)
    context['client'].create = checking_create
    assert call_responses(**context)[0]['status'] == 'completed'


def test_corrupt_cache_is_never_a_miss_or_new_request(context):
    context['cache_file'].parent.mkdir(parents=True)
    context['cache_file'].write_text('null')
    with pytest.raises(LiveTransportError, match='corrupt_cache_requires_reconciliation'):
        call_responses(**context)
    assert context['client'].calls == 0
    assert context['study_ledger'].summary()['logical_calls'] == 1
    assert context['cache_file'].read_text() == 'null'


def test_insufficient_study_budget_never_sends(context, tmp_path):
    limits = {'logical_calls': 1, 'physical_calls': 1, 'total_usd': '0',
              'arm_usd': {'terra_6_1': '0'}, 'stage_usd': {'pilot': '0'}}
    context['study_ledger'] = OfflineLedger(tmp_path / 'zero.sqlite3', limits, create=True)
    with pytest.raises(LiveTransportError, match='budget_limit'):
        call_responses(**context)
    assert context['client'].calls == 0
    assert json.loads(context['cache_file'].read_text())['status'] == 'blocked_before_send'


def test_no_usage_keeps_response_and_both_reservations_unknown(context):
    response = FakeResponse()
    response.usage = None
    response.model_dump = lambda **_: {'id': response.id, 'model': response.model, 'status': response.status}
    context['client'].response = response
    with pytest.raises(LiveTransportError, match='billing_usage_unknown'):
        call_responses(**context)
    saved = json.loads(context['cache_file'].read_text())
    assert saved['response']['id'] == 'response-fixture' and saved['reason'] == 'billing_usage_unknown'
    assert context['study_ledger'].summary()['unresolved_reservations'] == 1
    with context['campaign_ledger']._transaction() as db:
        assert db.execute('SELECT status FROM attempts').fetchone()[0] == 'unknown'


def test_stream_response_is_closed_and_recorded(context):
    class Event:
        type = 'response.completed'
        response = FakeResponse()
    class Stream:
        closed = False
        def __iter__(self):
            return iter([Event()])
        def close(self):
            self.closed = True
    stream = Stream()
    context['client'].response = stream
    context['request']['payload']['stream'] = True
    digest = hashlib.sha256(canonical_bytes(context['request'])).hexdigest()
    context['cache_file'] = context['cache_file'].with_name(digest + '.json')
    response, _trace = call_responses(**context)
    assert stream.closed and response['model'] == 'gpt-6.1-terra'
    assert response['output_text'] == FakeResponse.output_text
