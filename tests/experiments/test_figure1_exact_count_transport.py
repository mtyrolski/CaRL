"""Exact counter opt-in: fake SDK, isolated ledgers, and no network."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.figure1_pilot.budget import Figure1Ledger
from scripts.figure1_pilot.transport import LiveTransportError, call_responses
from carl.icl.campaign_budget import CampaignLimitExceeded
from carl.icl.reachability_data import canonical_bytes
from scripts.figure1_pilot.exact_token_preflight import project_count_request
from tests.experiments.test_figure1_transport import FakeClient, FakeResponse, context, no_network  # noqa: F401


class CountedResponse(FakeResponse):
    def model_dump(self, **kwargs):
        return {**super().model_dump(**kwargs), 'service_tier': 'default'}


class CountClient(FakeClient):
    def __init__(self):
        super().__init__(response=CountedResponse())
        self.input_tokens = self
        self.count_calls = 0
        self.count_value = 3
        self.count_error = None

    def count(self, **kwargs):
        self.count_calls += 1
        if self.count_error:
            raise self.count_error
        assert kwargs['model'] == 'gpt-6.1-terra' and kwargs['input'] == 'example'
        return {'object': 'response.input_tokens', 'input_tokens': self.count_value}


@pytest.fixture
def counted(context, tmp_path):  # noqa: F811
    previous = context['campaign_ledger']
    config = dict(previous.config)
    config.update(campaign_id='figure1_counted_test', ledger_path=str(tmp_path / 'counted.sqlite3'),
                  input_token_reservation={'mode': 'provider_exact_v1', 'max_input_tokens_per_count': 20,
                    'max_count_calls': 100, 'count_billing_assumption': 'conservative_model_input_rate_unverified'})
    path = tmp_path / 'counted.json'
    path.write_text(json.dumps(config))
    context['campaign_ledger'] = Figure1Ledger(path)
    context['client'] = CountClient()
    return context


def test_count_bound_and_generation_are_linked_and_cache_is_reused(counted):
    result, _trace = call_responses(**counted)
    cached, trace = call_responses(**counted)
    assert result == cached and trace['cache_hit']
    assert counted['client'].count_calls == counted['client'].calls == 1
    with counted['campaign_ledger']._transaction() as db:
        rows = db.execute('SELECT endpoint,status,charged_nanos FROM attempts ORDER BY rowid').fetchall()
        links = db.execute('SELECT count_attempt_id FROM figure1_generation_count_links').fetchall()
    assert rows[0] == ('responses.input_tokens', 'accounted_assumption', 6000)
    assert rows[1] == ('responses', 'settled', 5000)
    assert len(links) == 1 and links[0][0] == result['count_reservation_id']


def test_two_workers_send_one_count_and_one_generation(counted):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: call_responses(**counted), range(2)))
    assert counted['client'].count_calls == counted['client'].calls == 1
    assert sorted(trace['cache_hit'] for _, trace in results) == [False, True]


@pytest.mark.parametrize('error,value', [(TimeoutError('secret'), 3), (None, 21), (None, -1)])
def test_bad_or_unknown_count_keeps_reservation_and_no_generation(counted, error, value):
    counted['client'].count_error, counted['client'].count_value = error, value
    with pytest.raises(LiveTransportError, match='token_count_unknown'):
        call_responses(**counted)
    assert counted['client'].calls == 0 and counted['client'].count_calls == 1
    with counted['campaign_ledger']._transaction() as db:
        assert db.execute('SELECT status,charged_nanos FROM attempts').fetchone() == ('unknown', None)
    with pytest.raises(LiveTransportError, match='cached_error_requires_reconciliation'):
        call_responses(**counted)
    assert counted['client'].count_calls == 1


def test_usage_input_mismatch_stays_unknown(counted):
    counted['client'].count_value = 4
    with pytest.raises(LiveTransportError, match='billing_usage_unknown'):
        call_responses(**counted)
    with counted['campaign_ledger']._transaction() as db:
        assert db.execute('SELECT status FROM attempts ORDER BY rowid').fetchall() == [('accounted_assumption',), ('unknown',)]


def test_service_tier_missing_is_not_settled(counted):
    counted['client'].response = FakeResponse()
    with pytest.raises(LiveTransportError, match='unexpected_or_missing_service_tier'):
        call_responses(**counted)
    with counted['campaign_ledger']._transaction() as db:
        assert db.execute('SELECT status FROM attempts ORDER BY rowid').fetchall() == [('accounted_assumption',), ('unknown',)]


def test_token_count_bound_is_tied_to_payload_and_single_use(counted):
    call_responses(**counted)
    ledger = counted['campaign_ledger']
    with ledger._transaction() as db:
        count_id = db.execute('SELECT attempt_id FROM figure1_input_counts').fetchone()[0]
    changed = {**counted['request']['payload'], 'input': 'changed'}
    with pytest.raises(ValueError, match='mismatched'):
        ledger.quote_reservation('responses', changed, count_id=count_id)
    with pytest.raises(Exception):
        ledger.reserve('pilot', 'responses', counted['request']['payload'], counted['cache_file'].stem, count_id=count_id)


def test_reservation_cap_stops_before_counting(counted):
    with counted['campaign_ledger']._transaction() as db:
        db.execute("UPDATE metadata SET value='1' WHERE key='effective_limit_nanos'")
    with pytest.raises(LiveTransportError, match='budget_limit'):
        call_responses(**counted)
    assert counted['client'].count_calls == counted['client'].calls == 0


def test_parallel_count_reservations_share_hard_campaign_cap(counted):
    ledger = counted['campaign_ledger']
    with ledger._transaction() as db:
        db.execute("UPDATE metadata SET value='60000' WHERE key='effective_limit_nanos'")
    payload = counted['request']['payload']
    projected = project_count_request(payload)
    def reserve(index):
        try:
            return ledger.reserve_input_count('pilot', payload, projected, f'digest-{index}')
        except CampaignLimitExceeded:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(reserve, range(2)))
    assert sum(item is not None for item in ids) == 1
    with ledger._transaction() as db:
        assert db.execute('SELECT SUM(reserved_nanos),COUNT(*) FROM attempts').fetchone() == (40000, 1)


def test_output_over_limit_stays_unknown(counted):
    class TooMuchUsage:
        def model_dump(self, **_kwargs):
            return {'input_tokens': 3, 'output_tokens': 33}
    counted['client'].response.usage = TooMuchUsage()
    with pytest.raises(LiveTransportError, match='billing_usage_unknown'):
        call_responses(**counted)
    with counted['campaign_ledger']._transaction() as db:
        assert db.execute('SELECT status FROM attempts ORDER BY rowid').fetchall() == [('accounted_assumption',), ('unknown',)]


@pytest.mark.parametrize('unknown_kind', ['count', 'generation'])
def test_any_unknown_blocks_other_requests_until_reconciliation(counted, unknown_kind):
    if unknown_kind == 'count':
        counted['client'].count_error = TimeoutError('count timeout')
    else:
        counted['client'].error = TimeoutError('generation timeout')
    with pytest.raises(LiveTransportError):
        call_responses(**counted)
    counts_before, generations_before = counted['client'].count_calls, counted['client'].calls
    counted['client'].count_error = counted['client'].error = None
    request = {**counted['request'], 'slot': 1}
    digest = hashlib.sha256(canonical_bytes(request)).hexdigest()
    other = {**counted, 'request': request, 'cache_file': counted['cache_file'].parent / f'{digest}.json'}
    with pytest.raises(LiveTransportError, match='reconciliation_required'):
        call_responses(**other)
    assert counted['client'].count_calls == counts_before and counted['client'].calls == generations_before
