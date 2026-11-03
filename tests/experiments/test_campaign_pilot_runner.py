"""Pilot checkpoints are keyed, paired, resumable and fail closed."""
import csv
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carl.icl.api import RequestFailed, write_json
from carl.icl.config import ExperimentConfig, Limits
from carl.icl.search import SearchResult
from scripts.campaign_pilot_runner import (StopSignals, TrackingAPI, create_pilot_api, guard_scope,
                                          mark_interrupted_unknown, run_jobs, schedule, summarize_jobs, validate_freeze)
from scripts import freeze_small_pilot


def fixture_freeze() -> dict[str, Any]:
    return {'seed': 11, 'main_budget': 256, 'budgets': [128, 256, 512],
            'boards': [{'id': f'board_{i:03d}'} for i in range(100)]}


def test_schedule_pairs_methods_and_runs_main_budget_first() -> None:
    jobs = schedule(fixture_freeze(), 'freeze-hash')
    assert len(jobs) == 600 and len({job['key'] for job in jobs}) == 600
    assert [job['budget'] for job in jobs[::200]] == [256, 128, 512]
    assert [(job['board_id'], job['method']) for job in jobs[:4]] == [
        ('board_000', 'relative_u'), ('board_000', 'absolute_d'),
        ('board_001', 'relative_u'), ('board_001', 'absolute_d')]
    assert jobs[0]['key'] != schedule(fixture_freeze(), 'another-freeze')[0]['key']


def test_missing_freeze_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='freeze.json is required'):
        validate_freeze(tmp_path / 'freeze.json')


def test_no_freeze_is_published_when_audit_is_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / 'freeze.json'
    monkeypatch.setattr('sys.argv', ['freeze_small_pilot', '--output', str(target)])

    def incomplete(*_args: Any) -> None:
        raise ValueError('validation incomplete')

    monkeypatch.setattr(freeze_small_pilot, 'build_freeze', incomplete)
    with pytest.raises(ValueError, match='validation incomplete'):
        freeze_small_pilot.main()
    assert not target.exists()


def test_early_solved_stress_is_valid_but_full_budget_is_not_claimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = {'id': 'validation', 'state': ['#####', '#@$.#', '#####'], 'level_key': 'unused'}
    config = ExperimentConfig()
    plan = {'seed': 11, 'action_budget': 256, 'ledger_tag': 'validation', 'policy_fingerprint': 'fingerprint'}
    outcome = SearchResult(solved=True, finish_reason='solved', attempted_actions=1)
    case = {'provider': 'openai', 'seed': 11, 'instance': item, 'config': config.model_dump(mode='json'),
            'budget_tag': 'validation', 'policy_fingerprint': 'fingerprint', 'outcome': asdict(outcome),
            'verification': {'verified': True}, 'evidence_full_budget': False}
    (tmp_path / 'manifest.json').write_text(json.dumps(plan))
    (tmp_path / 'case.json').write_text(json.dumps(case))
    monkeypatch.setattr(freeze_small_pilot, 'prepare_plan', lambda _t, _s: (plan, config, item, {}))
    monkeypatch.setattr(freeze_small_pilot, 'load_trace',
                        lambda _p, _o: (object(), [{'event': 'finish', 'outcome': asdict(outcome)}]))
    monkeypatch.setattr(freeze_small_pilot, 'verify_tree', lambda *_args: {'verified': True})
    monkeypatch.setattr(freeze_small_pilot, 'verify_actions', lambda *_args: None)
    _plan, saved = freeze_small_pilot.audited_stress(tmp_path, tmp_path)
    assert saved['outcome']['solved'] and not saved['evidence_full_budget']


def test_interrupted_shard_resume_skips_only_completed_trials(tmp_path: Path) -> None:
    jobs = schedule(fixture_freeze(), 'freeze-hash')[:4]
    trials = tmp_path / 'trials'
    trials.mkdir()
    for job in jobs[:2]:
        write_json(trials / f"{job['key']}.json", {'job': job, 'status': 'complete'})
    observed: list[str] = []
    validated: list[str] = []

    def execute(job: dict[str, Any], _trace: Path) -> dict[str, Any]:
        observed.append(job['key'])
        return {'job': job, 'status': 'complete'}

    result = run_jobs(jobs, tmp_path, execute, resume=True,
                      validator=lambda job, _record, _trace: validated.append(job['key']))
    assert {k: result[k] for k in ('expected', 'completed', 'failed', 'missing')} == {
        'expected': 4, 'completed': 4, 'failed': 0, 'missing': 0}
    assert observed == [job['key'] for job in jobs[2:]]
    assert validated == [job['key'] for job in jobs[:2]]
    assert result['by_budget'][256]['paired_uncensored_boards'] == 2


def test_inflight_interruption_and_auth_failure_block_resume(tmp_path: Path) -> None:
    jobs = schedule(fixture_freeze(), 'freeze-hash')[:3]
    attempted: list[str] = []

    def interrupted(job: dict[str, Any], trace: Path) -> dict[str, Any]:
        attempted.append(job['key'])
        trace.write_text('partial action trace\n')
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_jobs(jobs, tmp_path, interrupted, resume=False)
    progress = summarize_jobs(jobs, tmp_path)
    assert {k: progress[k] for k in ('expected', 'completed', 'failed', 'missing')} == {
        'expected': 3, 'completed': 0, 'failed': 1, 'missing': 2}
    with pytest.raises(ValueError, match='requires inspection'):
        run_jobs(jobs, tmp_path, interrupted, resume=True)
    assert attempted == [jobs[0]['key']]
    assert (tmp_path / 'trials' / f"{jobs[0]['key']}.trace.jsonl").read_text() == 'partial action trace\n'

    auth_dir = tmp_path / 'auth'
    calls: list[str] = []

    def auth_error(job: dict[str, Any], _trace: Path) -> dict[str, Any]:
        calls.append(job['key'])
        return {'job': job, 'status': 'failed', 'failure_category': 'provider_or_transport_error'}

    result = run_jobs(jobs, auth_dir, auth_error, resume=False)
    assert {k: result[k] for k in ('expected', 'completed', 'failed', 'missing')} == {
        'expected': 3, 'completed': 0, 'failed': 1, 'missing': 2}
    assert calls == [jobs[0]['key']]
    progress = json.loads((auth_dir / 'progress.json').read_text())
    assert progress['failed'] == 1


def test_keyboard_interrupt_marks_only_own_inflight_reservation_unknown(tmp_path: Path) -> None:
    db_path = tmp_path / 'ledger.sqlite'
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE attempts (id TEXT,stage TEXT,tag TEXT,status TEXT,reserved_nanos INTEGER)')
        db.executemany('INSERT INTO attempts VALUES (?,?,?,?,?)', [
            ('mine', 'pilot_relative', 'trial', 'reserved', 100),
            ('other', 'pilot_relative', 'other', 'reserved', 200),
        ])

    def mark_unknown(identifier: str) -> None:
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE attempts SET status='unknown' WHERE id=?", (identifier,))

    api: Any = SimpleNamespace(ledger=SimpleNamespace(path=db_path, mark_unknown=mark_unknown), budget_tag='trial')
    mark_interrupted_unknown('relative_u', api)
    with sqlite3.connect(db_path) as db:
        assert db.execute('SELECT id,status,reserved_nanos FROM attempts ORDER BY id').fetchall() == [
            ('mine', 'unknown', 100), ('other', 'reserved', 200)]


def test_guard_scope_only_skips_stage_if_global_headroom_is_proven(tmp_path: Path) -> None:
    db_path = tmp_path / 'ledger.sqlite'
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE metadata (key TEXT,value TEXT)')
        db.execute("INSERT INTO metadata VALUES ('effective_limit_nanos','40000000000')")
        db.execute('CREATE TABLE attempts (stage TEXT,charged_nanos INTEGER,reserved_nanos INTEGER)')
        db.execute("INSERT INTO attempts VALUES ('pilot_relative', 11900000000, 0)")
    config = {'models': {'gpt-6-luna': {'cache_write_12h_usd_per_million': '0.25',
              'max_input_tokens': 1050000, 'output_usd_per_million': '0.5', 'regional_multiplier': '1.1'}},
              'stage_limits_usd': {'pilot_relative': '12'}}
    api: Any = SimpleNamespace(ledger=SimpleNamespace(path=db_path, config=config), last_output_cap=4096)
    assert guard_scope('relative_u', api) == 'method_stage'
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO attempts VALUES ('other', 28000000000, 0)")
    assert guard_scope('relative_u', api) == 'campaign'


def test_exhausted_stage_finishes_other_method_only_in_current_phase(tmp_path: Path) -> None:
    jobs = schedule(fixture_freeze(), 'freeze-hash')[:4] + schedule(fixture_freeze(), 'freeze-hash')[200:202]
    calls: list[tuple[int, str, str]] = []

    def execute(job: dict[str, Any], _trace: Path) -> dict[str, Any]:
        calls.append((job['budget'], job['board_id'], job['method']))
        if job['method'] == 'relative_u':
            return {'job': job, 'status': 'failed', 'failure_category': 'campaign_stage_guard',
                    'budget_scope': 'method_stage'}
        return {'job': job, 'status': 'complete', 'completion_class': 'action_budget'}

    result = run_jobs(jobs, tmp_path, execute, resume=False)
    assert calls == [(256, 'board_000', 'relative_u'), (256, 'board_000', 'absolute_d'),
                     (256, 'board_001', 'absolute_d')]
    assert result['failed'] == 1 and result['completed'] == 2 and result['missing'] == 3
    assert result['by_budget'][256]['family'] == 'real_12x12x4'
    assert result['by_budget'][256]['methods']['relative_u']['failed'] == 1
    assert result['by_budget'][256]['methods']['relative_u']['missing'] == 1
    assert result['by_budget'][256]['methods']['absolute_d']['completed'] == 2
    assert result['by_budget'][256]['paired_uncensored_boards'] == 0
    with (tmp_path / 'summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    relative = next(row for row in rows if row['budget'] == '256' and row['method'] == 'relative_u')
    assert relative['failed'] == '1' and relative['missing'] == '1' and relative['paired_uncensored_boards'] == '0'
    assert all(call[0] == 256 for call in calls)
    # A resume still cannot promote a partially completed B* phase.
    assert run_jobs(jobs, tmp_path, execute, resume=True)['missing'] == 3


def test_local_limit_is_explicit_and_stops_next_budget_phase(tmp_path: Path) -> None:
    all_jobs = schedule(fixture_freeze(), 'freeze-hash')
    jobs = all_jobs[:2] + all_jobs[200:202]

    def execute(job: dict[str, Any], _trace: Path) -> dict[str, Any]:
        return {'job': job, 'status': 'complete', 'completion_class': 'local_limit_before_B'}

    result = run_jobs(jobs, tmp_path, execute, resume=False)
    assert result['completed'] == 2 and result['local_limit_before_B'] == 2 and result['missing'] == 2
    assert result['by_budget'][256]['methods']['relative_u']['censored'] == 1
    assert result['by_budget'][256]['paired_uncensored_boards'] == 0
    with (tmp_path / 'summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert next(row for row in rows if row['budget'] == '256')['censored'] == '1'


def test_workers_two_resume_without_duplicates_or_phase_overlap(tmp_path: Path) -> None:
    all_jobs = schedule(fixture_freeze(), 'freeze-hash')
    jobs = all_jobs[:4] + all_jobs[200:204]
    barrier = threading.Barrier(2, timeout=5)
    lock = threading.Lock()
    started: list[tuple[int, str]] = []
    finished_first: set[str] = set()

    def execute(job: dict[str, Any], _trace: Path) -> dict[str, Any]:
        with lock:
            if job['budget'] != 256:
                assert len(finished_first) == 4
            started.append((job['budget'], job['key']))
        barrier.wait()
        with lock:
            if job['budget'] == 256:
                finished_first.add(job['key'])
        return {'job': job, 'status': 'complete', 'completion_class': 'action_budget'}

    result = run_jobs(jobs, tmp_path, execute, resume=False, workers=2)
    assert result['completed'] == 8 and result['failed'] == result['missing'] == 0
    assert len(started) == len(set(started)) == 8
    assert run_jobs(jobs, tmp_path, execute, resume=True, workers=2)['completed'] == 8
    assert len(started) == 8


def test_workers_two_auth_failure_stops_scheduling_and_drains_inflight(tmp_path: Path) -> None:
    jobs = schedule(fixture_freeze(), 'freeze-hash')[:6]
    signals = StopSignals()
    barrier = threading.Barrier(2, timeout=5)
    started: list[str] = []
    lock = threading.Lock()

    def execute(job: dict[str, Any], _trace: Path) -> dict[str, Any]:
        with lock:
            started.append(job['key'])
        barrier.wait()
        if job['method'] == 'relative_u':
            return {'job': job, 'status': 'failed', 'failure_category': 'provider_or_transport_error'}
        assert signals.global_stop.wait(5)
        return {'job': job, 'status': 'failed', 'failure_category': 'peer_cancelled'}

    result = run_jobs(jobs, tmp_path, execute, resume=False, workers=2, signals=signals)
    assert len(started) == 2 and len(set(started)) == 2
    assert result['failed'] == 2 and result['missing'] == 4 and result['in_flight'] == 0


@pytest.mark.parametrize('interrupt_at', ['progress', 'second_start'])
def test_workers_two_interrupt_outside_wait_stops_and_drains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             interrupt_at: str) -> None:
    from scripts import campaign_pilot_runner as runner  # pylint: disable=import-outside-toplevel

    jobs = schedule(fixture_freeze(), 'freeze-hash')[:4]
    signals = StopSignals()
    entered = threading.Event()
    original_write = runner.write_json
    original_progress = runner.write_progress
    interrupted = False

    def write_guard(path: Path, value: Any) -> None:
        nonlocal interrupted
        if interrupt_at == 'second_start' and path.name == f"{jobs[1]['key']}.started.json" and not interrupted:
            assert entered.wait(5)
            interrupted = True
            raise KeyboardInterrupt
        original_write(path, value)

    def progress_guard(output: Path, value: dict[str, Any]) -> None:
        nonlocal interrupted
        if interrupt_at == 'progress' and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        original_progress(output, value)

    monkeypatch.setattr(runner, 'write_json', write_guard)
    monkeypatch.setattr(runner, 'write_progress', progress_guard)

    def execute(job: dict[str, Any], _trace: Path) -> dict[str, Any]:
        entered.set()
        if interrupt_at == 'second_start' or job['method'] == 'absolute_d':
            assert signals.global_stop.wait(5)
            return {'job': job, 'status': 'failed', 'failure_category': 'peer_cancelled'}
        return {'job': job, 'status': 'complete', 'completion_class': 'action_budget'}

    with pytest.raises(KeyboardInterrupt):
        run_jobs(jobs, tmp_path, execute, resume=False, workers=2, signals=signals)
    assert signals.global_stop.is_set() and interrupted
    result = summarize_jobs(jobs, tmp_path)
    assert result['in_flight'] == 0 and result['completed'] + result['failed'] >= 1
    assert not (tmp_path / 'trials' / f"{jobs[2]['key']}.started.json").exists()


def test_workers_two_stage_stop_continues_other_method_without_next_phase(tmp_path: Path) -> None:
    all_jobs = schedule(fixture_freeze(), 'freeze-hash')
    jobs = all_jobs[:4] + all_jobs[200:202]
    signals = StopSignals()
    barrier = threading.Barrier(2, timeout=5)
    started: list[tuple[int, str, str]] = []
    lock = threading.Lock()

    def execute(job: dict[str, Any], _trace: Path) -> dict[str, Any]:
        with lock:
            started.append((job['budget'], job['board_id'], job['method']))
        if job['board_id'] == 'board_000':
            barrier.wait()
        if job['method'] == 'relative_u':
            signals.method_stop['relative_u'].set()
            return {'job': job, 'status': 'failed', 'failure_category': 'campaign_stage_guard',
                    'budget_scope': 'method_stage'}
        assert signals.method_stop['relative_u'].wait(5)
        return {'job': job, 'status': 'complete', 'completion_class': 'action_budget'}

    result = run_jobs(jobs, tmp_path, execute, resume=False, workers=2, signals=signals)
    assert result['failed'] == 1 and result['completed'] == 2 and result['missing'] == 3
    assert len(started) == 3 and all(budget == 256 for budget, _, _ in started)
    assert sum(method == 'relative_u' for _, _, method in started) == 1


def test_stage_is_bound_per_client_under_forced_interleaving(monkeypatch: pytest.MonkeyPatch) -> None:
    barrier = threading.Barrier(2, timeout=5)

    class FakeAPI:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.stage = 'unassigned'
            barrier.wait()

    monkeypatch.setattr('scripts.campaign_pilot_runner.TrackingAPI', FakeAPI)
    config = ExperimentConfig()
    freeze = {'max_generator_calls': 10, 'max_total_tokens': 1000}
    signals = StopSignals()
    with ThreadPoolExecutor(max_workers=2) as pool:
        clients = list(pool.map(lambda method: create_pilot_api(config, freeze, method, signals),
                                ('relative_u', 'absolute_d')))
    assert [client.stage for client in clients] == ['pilot_relative', 'pilot_absolute']


class FakeLedger:
    """Capture separate charged attempts without touching the actual campaign."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.reservations: list[tuple[Any, ...]] = []
        self.unknown: list[str] = []

    def event(self, *args: Any) -> None:
        self.events.append(args)

    def reserve(self, *args: Any) -> str:
        self.reservations.append(args)
        return str(len(self.reservations))

    def settle(self, _reservation: str, usage: dict[str, Any], _payload: dict[str, Any]) -> bool:
        return bool(usage)

    def mark_unknown(self, reservation: str) -> None:
        self.unknown.append(reservation)


class FakeHTTPError(Exception):
    """A provider error with an explicit HTTP status."""

    def __init__(self, status: int) -> None:
        super().__init__('provider failure')
        self.status_code = status


@pytest.mark.parametrize('status', [429, 500, 503])
def test_transient_http_has_separate_budgeted_retry_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    ledger = FakeLedger()
    monkeypatch.setattr('carl.icl.api._campaign_ledger', lambda: ledger)
    monkeypatch.setattr('scripts.campaign_pilot_runner.time.sleep', lambda _seconds: None)
    requests: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        requests.append(kwargs)
        if len(requests) == 1:
            raise FakeHTTPError(status)
        return SimpleNamespace(status='completed', output_text='{"ok":true}', output=[], model='test-model',
                               id='r1', usage=SimpleNamespace(model_dump=lambda: {'input_tokens': 10, 'output_tokens': 2}),
                               model_dump=lambda **_kwargs: {'status': 'completed'})

    client = SimpleNamespace(responses=SimpleNamespace(create=create), max_retries=0)
    api = TrackingAPI(tmp_path, 'openai', Limits(max_requests=5, max_output_tokens=1024), client,
                      max_generator_calls=5, max_total_tokens=100000)
    payload = {'model': 'test-model', 'input': 'same', 'max_output_tokens': 64, 'metadata': {'component': 'policy'}}
    assert api.call('responses', payload)['output_text'] == '{"ok":true}'
    assert requests == [payload, payload]
    assert len(ledger.reservations) == 2 and ledger.unknown == ['1']
    records = [json.loads(path.read_text()) for path in (tmp_path / 'openai').glob('*.json')]
    assert len(records) == 2 and {record['status'] for record in records} == {'error', 'completed'}
    assert {record.get('cache_attempt_slot', 0) for record in records} == {0, 2}
    assert all(record['request']['payload'] == payload for record in records)
    assert sum(event[1] == 'http_transient_retry' for event in ledger.events) == 1
    assert not any(event[1] == 'json_retry' for event in ledger.events)


@pytest.mark.parametrize('status', [401, 403, 404])
def test_auth_and_nontransient_http_do_not_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    ledger = FakeLedger()
    monkeypatch.setattr('carl.icl.api._campaign_ledger', lambda: ledger)
    calls: list[int] = []

    def create(**_kwargs: Any) -> None:
        calls.append(status)
        raise FakeHTTPError(status)

    client = SimpleNamespace(responses=SimpleNamespace(create=create), max_retries=0)
    api = TrackingAPI(tmp_path, 'openai', Limits(max_requests=5, max_output_tokens=1024), client,
                      max_generator_calls=5, max_total_tokens=100000)
    with pytest.raises(RequestFailed):
        api.call('responses', {'model': 'test-model', 'input': 'same', 'max_output_tokens': 64})
    assert calls == [status] and len(ledger.reservations) == 1 and ledger.unknown == ['1']


def test_transient_retry_is_bounded_and_unknown_transport_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = FakeLedger()
    monkeypatch.setattr('carl.icl.api._campaign_ledger', lambda: ledger)
    monkeypatch.setattr('scripts.campaign_pilot_runner.time.sleep', lambda _seconds: None)
    calls: list[str] = []

    def always_limited(**_kwargs: Any) -> None:
        calls.append('request')
        raise FakeHTTPError(429)

    client = SimpleNamespace(responses=SimpleNamespace(create=always_limited), max_retries=0)
    api = TrackingAPI(tmp_path / 'rate', 'openai', Limits(max_requests=5, max_output_tokens=1024), client,
                      max_generator_calls=5, max_total_tokens=100000)
    payload = {'model': 'test-model', 'input': 'same', 'max_output_tokens': 64}
    with pytest.raises(RequestFailed):
        api.call('responses', payload)
    assert len(calls) == 2 and ledger.unknown == ['1', '2']
    assert len(list((tmp_path / 'rate' / 'openai').glob('*.json'))) == 2

    def disconnected(**_kwargs: Any) -> None:
        calls.append('transport')
        raise ConnectionError('unknown billing')

    client = SimpleNamespace(responses=SimpleNamespace(create=disconnected), max_retries=0)
    api = TrackingAPI(tmp_path / 'transport', 'openai', Limits(max_requests=5, max_output_tokens=1024), client,
                      max_generator_calls=5, max_total_tokens=100000)
    with pytest.raises(RequestFailed):
        api.call('responses', payload)
    assert calls.count('transport') == 1


def test_global_stop_blocks_the_next_request_without_a_new_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = FakeLedger()
    monkeypatch.setattr('carl.icl.api._campaign_ledger', lambda: ledger)
    signals = StopSignals()
    signals.global_stop.set()
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: None), max_retries=0)
    api = TrackingAPI(tmp_path, 'openai', Limits(), client, max_generator_calls=5, max_total_tokens=100000,
                      method='absolute_d', signals=signals)
    with pytest.raises(RequestFailed, match='other worker error'):
        api.call('responses', {'model': 'test-model', 'input': 'same', 'max_output_tokens': 64})
    assert not ledger.reservations and api.cancel_scope == 'campaign_or_peer_error'
