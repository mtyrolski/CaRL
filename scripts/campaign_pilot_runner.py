"""Freeze-gated, paired and checkpointed small-board pilot execution.

This module never creates a freeze and cannot turn a cached query into a
completed solver run. A partially started case remains blocked for review.
"""
import hashlib
import csv
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from decimal import ROUND_CEILING, Decimal
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Any, Iterator, cast

from carl.icl.api import BudgetExceeded, RequestFailed, _CACHE_ATTEMPT_SLOT, write_json
from carl.icl.boards import canonical_layout_id, from_rows, state_hash
from carl.icl.budget_sweep import CODE_FILES, _sweep_lock
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.campaign_budget import CampaignLedger
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import calibration_path, components_for_run, credentials
from carl.icl.search import SearchResult, search
from carl.icl.search_tree import SearchTreeRecorder, TreeNode
from scripts.campaign_scale_stress import DurableEnvironment, DurableTree, verify_actions
from scripts.estimate_campaign_pilot import ledger_snapshot
from scripts.tune_campaign_scales import CappedAPI, failure_category, verify_tree

METHODS = ('relative_u', 'absolute_d')
STAGES = {'relative_u': 'pilot_relative', 'absolute_d': 'pilot_absolute'}
EXTRA_CODE = ('scripts/campaign_pilot_runner.py', 'scripts/run_budget_sweep.py',
              'scripts/tune_campaign_scales.py', 'scripts/campaign_scale_stress.py',
              'scripts/estimate_campaign_pilot.py', 'carl/icl/campaign_budget.py',
              'carl/icl/search_tree.py')
HTTP_RETRY_POLICY = {'max_retries': 1, 'backoff_seconds': 0.5, 'statuses': ['429', '500-599'],
                     'backoff': 'exponential', 'unknown_reservation': 'retain', 'slots': 'original+2'}


class StopSignals:
    """Prevent new calls after a global error or method-specific budget stop."""

    def __init__(self) -> None:
        self.global_stop = threading.Event()
        self.method_stop = {method: threading.Event() for method in METHODS}


class RetryEventLedger:
    """Label separate HTTP retry slots without changing the shared ledger."""

    def __init__(self, delegate: CampaignLedger) -> None:
        self.delegate = delegate

    def event(self, stage: str, kind: str, endpoint: str, digest_value: str, tag: str = '') -> None:
        if kind == 'json_retry' and _CACHE_ATTEMPT_SLOT.get() in (2, 3):
            kind = 'http_transient_retry'
        self.delegate.event(stage, kind, endpoint, digest_value, tag)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class TrackingAPI(CappedAPI):
    """Track guard context and retry only explicit HTTP 429/5xx in isolated slots."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.signals: StopSignals | None = kwargs.pop('signals', None)
        self.method: str | None = kwargs.pop('method', None)
        super().__init__(*args, **kwargs)
        self.last_output_cap: int | None = None
        self.cancel_scope: str | None = None
        if self.ledger is not None:
            self.ledger = cast(CampaignLedger, RetryEventLedger(self.ledger))

    def check_stop(self) -> None:
        if self.signals is not None and self.signals.global_stop.is_set():
            self.cancel_scope = 'campaign_or_peer_error'
            raise RequestFailed('Pilot stopped after a campaign or other worker error')
        if self.signals is not None and self.method is not None and self.signals.method_stop[self.method].is_set():
            self.cancel_scope = 'method_stage'
            raise RequestFailed('Pilot method stopped after its stage limit')

    def signal_guard(self, error: BudgetExceeded) -> None:
        if self.signals is None or 'Campaign or stage ' not in str(error):
            return
        try:
            scope = guard_scope(self.method, self) if self.method is not None else 'uncertain'
        except Exception:  # Fail closed when the reason cannot be determined.
            scope = 'uncertain'
        self.cancel_scope = scope
        if scope == 'method_stage' and self.method is not None:
            self.signals.method_stop[self.method].set()
        else:
            self.signals.global_stop.set()

    def call(self, endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
        self.check_stop()
        maximum = payload.get('max_output_tokens')
        self.last_output_cap = maximum if isinstance(maximum, int) and not isinstance(maximum, bool) else None
        try:
            return super().call(endpoint, payload, mock)
        except BudgetExceeded as error:
            self.signal_guard(error)
            raise
        except RequestFailed:
            slot = _CACHE_ATTEMPT_SLOT.get()
            if self.provider != 'openai' or slot not in (0, 1):
                if self.signals is not None:
                    self.signals.global_stop.set()
                raise
            request = {'endpoint': endpoint, 'payload': payload}
            key = request if slot == 0 else {'request': request, 'cache_attempt_slot': slot}
            request_hash = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
            path = self.cache_dir / f'{request_hash}.json'
            if not path.is_file():
                if self.signals is not None:
                    self.signals.global_stop.set()
                raise
            record = json.loads(path.read_text())
            status = record.get('http_status')
            if (record.get('status') != 'error' or record.get('request') != request
                    or not isinstance(status, int) or isinstance(status, bool)
                    or not (status == 429 or 500 <= status <= 599)):
                if self.signals is not None:
                    self.signals.global_stop.set()
                raise
            # Preserve the failed response and its possibly unknown reservation.
            # The second slot is distinct for the original JSON attempt and for
            # the optional JSON-syntax retry, but transmits the identical payload.
            time.sleep(HTTP_RETRY_POLICY['backoff_seconds'] * (2 ** 0))
            self.check_stop()
            token = _CACHE_ATTEMPT_SLOT.set(slot + 2)
            try:
                try:
                    return super().call(endpoint, payload, mock)
                except BudgetExceeded as error:
                    self.signal_guard(error)
                    raise
                except RequestFailed:
                    if self.signals is not None:
                        self.signals.global_stop.set()
                    raise
            finally:
                _CACHE_ATTEMPT_SLOT.reset(token)


def guard_scope(method: str, api: TrackingAPI) -> str:
    """Distinguish a method-stage stop only when the ledger proves it unambiguously."""
    if api.ledger is None or api.last_output_cap is None:
        return 'uncertain'
    with sqlite3.connect(api.ledger.path) as db:
        db.execute('BEGIN')
        total_spend = int(db.execute('SELECT COALESCE(SUM(COALESCE(charged_nanos,reserved_nanos)),0) FROM attempts').fetchone()[0])
        stage_spend = int(db.execute('SELECT COALESCE(SUM(COALESCE(charged_nanos,reserved_nanos)),0) '
                                     'FROM attempts WHERE stage=?', (STAGES[method],)).fetchone()[0])
        effective = int(db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()[0])
    price = api.ledger.config['models']['gpt-6-luna']
    upper = (Decimal(str(price['cache_write_12h_usd_per_million'])) * 2 * price['max_input_tokens']
             + Decimal(str(price['output_usd_per_million'])) * Decimal('1.5') * api.last_output_cap)
    reserve_nanos = int((upper * Decimal(str(price['regional_multiplier'])) * 1000).to_integral_value(rounding=ROUND_CEILING))
    stage_remaining = int(Decimal(str(api.ledger.config['stage_limits_usd'][STAGES[method]])) * 1_000_000_000) - stage_spend
    global_remaining = effective - total_spend
    if global_remaining < reserve_nanos:
        return 'campaign'
    if stage_remaining < reserve_nanos:
        return 'method_stage'
    return 'uncertain'


def mark_interrupted_unknown(method: str, api: TrackingAPI) -> None:
    """Keep this trial's in-flight reservations charged when interrupted."""
    if api.ledger is None:
        return
    with sqlite3.connect(api.ledger.path) as db:
        pending = db.execute("SELECT id FROM attempts WHERE stage=? AND tag=? AND status='reserved'",
                             (STAGES[method], api.budget_tag)).fetchall()
    for (reservation,) in pending:
        api.ledger.mark_unknown(reservation)


def digest(path: Path) -> str:
    """Return a file hash without revealing file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path: Path) -> str:
    """Reject inputs outside the repository and return a canonical relative path."""
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError('Freeze input is outside the repository')
    return resolved.relative_to(ROOT.resolve()).as_posix()


def trial_key(freeze_hash: str, board_id: str, method: str, seed: int, budget: int) -> str:
    """Bind the durable checkpoint to all independent run identifiers."""
    value = [freeze_hash, board_id, method, seed, budget]
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def schedule(freeze: dict[str, Any], freeze_hash: str) -> list[dict[str, Any]]:
    """Run B* first, then each other budget; pair methods on each board."""
    budgets = [freeze['main_budget'], *(b for b in freeze['budgets'] if b != freeze['main_budget'])]
    return [{'key': trial_key(freeze_hash, board['id'], method, freeze['seed'], budget),
             'freeze_hash': freeze_hash, 'board_id': board['id'], 'method': method,
             'seed': freeze['seed'], 'budget': budget}
            for budget in budgets for board in freeze['boards'] for method in METHODS]


def validate_freeze(path: Path) -> tuple[dict[str, Any], str, dict[str, ExperimentConfig], Any, list[Any]]:
    """Validate the full real-data, model, calibration and code contract without API calls."""
    if not path.is_file() or path.name != 'freeze.json':
        raise ValueError('A finalized freeze.json is required before planning or running the pilot')
    freeze = json.loads(path.read_text(encoding='utf-8'))
    if (freeze.get('version') != 1 or freeze.get('status') != 'frozen'
            or freeze.get('purpose') != 'small_board_pilot' or freeze.get('family') != 'real_12x12x4'
            or freeze.get('split') != 'pilot100'):
        raise ValueError('Freeze is not a finalized small-board pilot contract')
    budgets, seed, main = freeze.get('budgets'), freeze.get('seed'), freeze.get('main_budget')
    if (not isinstance(budgets, list) or len(budgets) != 3
            or any(not isinstance(b, int) or isinstance(b, bool) or b <= 0 for b in budgets)
            or budgets != sorted(set(budgets)) or not isinstance(main, int) or isinstance(main, bool) or main not in budgets
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError('Freeze must specify three increasing budgets, their main budget and one seed')
    if any(not isinstance(freeze.get(k), int) or isinstance(freeze[k], bool) or freeze[k] <= 0
           for k in ('max_generator_calls', 'max_total_tokens')):
        raise ValueError('Freeze must provide positive logical generator and total-token limits')
    if freeze.get('http_retry_policy') != HTTP_RETRY_POLICY:
        raise ValueError('Freeze must bind the exact bounded transient HTTP retry policy')
    conditions = freeze.get('conditions')
    if not isinstance(conditions, dict) or set(conditions) != set(METHODS):
        raise ValueError('Exactly relative_u and absolute_d must be frozen')
    configs: dict[str, ExperimentConfig] = {}
    for method in METHODS:
        entry = conditions[method]
        config = ExperimentConfig.model_validate(entry['config'])
        if (config.model != 'gpt-6-luna' or config.family != freeze['family'] or config.split != 'pilot100'
                or config.method != method or config.instances != 100 or config.easy or config.repeat_seeds != [seed]
                or config.action_budget != main):
            raise ValueError(f'Invalid frozen condition for {method}')
        configs[method] = config
    ignored = {'name', 'method', 'scales'}
    if configs[METHODS[0]].model_dump(exclude=ignored) != configs[METHODS[1]].model_dump(exclude=ignored):
        raise ValueError('Both conditions must have identical non-label settings and available limits')
    training, items = load_data(configs[METHODS[0]])
    _other_training, other_items = load_data(configs[METHODS[1]])
    if items != other_items or len(items) != 100:
        raise ValueError('Frozen methods must load the same 100 initial states')
    boards = freeze.get('boards')
    expected = [{'id': item['id'], 'state_hash': state_hash(from_rows(item['state'])),
                 'layout_id': canonical_layout_id(from_rows(item['state']))} for item in items]
    if boards != expected or len({row['id'] for row in expected}) != 100 or len({row['layout_id'] for row in expected}) != 100:
        raise ValueError('Freeze board identities do not match the 100 unique full pilot layouts')
    source_hashes = freeze.get('source_sha256')
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError('Freeze must contain source and data hashes')
    root = configs[METHODS[0]].path('data_dir')
    audit = freeze.get('audit')
    if not isinstance(audit, dict) or set(audit) != {'tuning_dir', 'stress_dir', 'data_dir', 'integrity_report'}:
        raise ValueError('Freeze must identify its validation, stress and cohort integrity audits')
    tuning, stress = ROOT / audit['tuning_dir'], ROOT / audit['stress_dir']
    integrity_path = ROOT / audit['integrity_report']
    if any(relative_path(path) != audit[key] for path, key in (
        (tuning, 'tuning_dir'), (stress, 'stress_dir'), (root, 'data_dir'), (integrity_path, 'integrity_report'))):
        raise ValueError('Freeze audit locations do not match the selected data')
    required = set(CODE_FILES) | set(EXTRA_CODE) | {relative_path(root / 'training.json'),
                relative_path(root / 'campaign_dataset_manifest.json'),
                relative_path(root / freeze['family'] / 'pilot100.json'),
                relative_path(integrity_path), relative_path(tuning / 'manifest.json'),
                relative_path(tuning / 'summary.json'), relative_path(stress / 'manifest.json'),
                relative_path(stress / 'case.json'), relative_path(stress / 'trace.jsonl'),
                'scripts/freeze_small_pilot.py', 'scripts/campaign_real12_stress.py',
                'scripts/tune_real12_campaign_scales.py', 'scripts/estimate_real12_pilot.py',
                'configs/icl/campaign_budget.json'}
    required.update(relative_path(item) for item in tuning.glob('tune_*.json'))
    if configs[METHODS[0]].value_examples == 'trajectory':
        required.add(relative_path(root / 'value_examples.json'))
    for method in METHODS:
        component = components_for_run(configs[method], 'replay', training, from_rows(items[0]['state']))
        entry = conditions[method]
        if (entry['policy_fingerprint'] != component.fingerprint() or component.calibrated is None
                or entry['calibration_signature'] != component.calibrated['signature']):
            raise ValueError(f'Frozen policy or calibration differs for {method}')
        required.add(relative_path(calibration_path(configs[method], component)))
    if not required <= set(source_hashes):
        raise ValueError('Freeze omits a required code, configuration, calibration or data hash')
    for name, expected_hash in source_hashes.items():
        item = ROOT / name
        if relative_path(item) != name or not item.is_file() or digest(item) != expected_hash:
            raise ValueError(f'Frozen source hash mismatch: {name}')
    from scripts.campaign_real12_stress import checked_sources  # pylint: disable=import-outside-toplevel
    from scripts.freeze_small_pilot import audited_stress  # pylint: disable=import-outside-toplevel

    tuning_manifest, summary, _cases = checked_sources(tuning)
    stress_plan, _stress_case = audited_stress(tuning, stress)
    if (freeze['budgets'] != tuning_manifest['design']['pilot_budget_candidates']
            or freeze['main_budget'] != tuning_manifest['design']['candidate_main_budget']
            or freeze['main_budget'] != stress_plan['action_budget']
            or freeze['seed'] != tuning_manifest['seed']
            or any(configs[method].scales != summary['methods'][method]['selected_pair'] for method in METHODS)):
        raise ValueError('Freeze scales or budgets differ from audited validation')
    integrity = json.loads(integrity_path.read_text())
    if (integrity.get('boards') != 100 or integrity.get('unique_layouts') != 100
            or integrity.get('witnesses_replayed') != 100 or not integrity.get('every_witness_solved')
            or any(integrity.get('overlap_by_existing_split', {}).values())
            or integrity.get('overlap_observed_requests_or_history') != 0
            or digest(root / freeze['family'] / 'pilot100.json') != integrity.get('public_manifest_sha256')):
        raise ValueError('Freeze cohort integrity audit is invalid')
    return freeze, digest(path), configs, training, items


def cost_plan(freeze: dict[str, Any], configs: dict[str, ExperimentConfig], ledger: dict[str, Any],
              pricing: dict[str, Any]) -> dict[str, Any]:
    """Compute deliberately conservative configured per-run token caps, not a forecast."""
    rates = pricing['models']['gpt-6-luna']
    maximum = {}
    for method, config in configs.items():
        incoming = min(config.limits.max_total_input_tokens, freeze['max_total_tokens'])
        outgoing = min(config.limits.max_output_tokens, freeze['max_total_tokens'])
        maximum[method] = ((incoming * float(rates['cache_write_12h_usd_per_million'])
                            + outgoing * float(rates['output_usd_per_million']))
                           * float(rates['regional_multiplier']) / 1_000_000)
    return {'per_run_conservative_token_cap_usd': maximum,
            'all_600_runs_conservative_token_cap_usd': sum(maximum.values()) * 300,
            'campaign_limit_usd': ledger['effective_limit_usd'], 'campaign_committed_usd': ledger['committed_usd'],
            'campaign_remaining_usd': ledger['remaining_usd'],
            'method_stage_remaining_usd': {m: ledger['stages'][STAGES[m]]['remaining_usd'] for m in METHODS},
            'max_additional_ledger_commitment_usd': max(0.0, min(ledger['remaining_usd'], sum(
                max(0.0, ledger['stages'][STAGES[m]]['remaining_usd']) for m in METHODS))),
            'note': 'Token-cap sum is a configured token-usage scenario, not an expected cost, invoice, or bound '
                    'on unknown provider charges. The separate ledger commitment ceiling includes reservations '
                    'and enforces the lower global and per-stage limits.'}


def summarize_jobs(jobs: list[dict[str, Any]], output: Path, active: set[str] | None = None) -> dict[str, Any]:
    """Count only complete, durable per-run records; cache files are never checkpoints."""
    done = failed = in_flight = 0
    active = active or set()
    classes = {'solved': 0, 'action_budget': 0, 'local_limit_before_B': 0, 'frontier_exhausted': 0,
               'other_completed': 0}
    per_budget: dict[int, dict[str, Any]] = {}
    eligible: set[tuple[int, str, str]] = set()
    solved: set[tuple[int, str, str]] = set()
    for job in jobs:
        budget = job['budget']
        group = per_budget.setdefault(budget, {'family': 'real_12x12x4',
                                               'methods': {m: {'expected': 0, 'completed': 0, 'failed': 0, 'in_flight': 0,
                                                                 'missing': 0, 'censored': 0, 'solved': 0}
                                                          for m in METHODS}, 'paired_uncensored_boards': 0})
        counts = group['methods'][job['method']]
        counts['expected'] += 1
        case = output / 'trials' / f"{job['key']}.json"
        started = output / 'trials' / f"{job['key']}.started.json"
        if case.exists():
            record = json.loads(case.read_text())
            if record.get('job') != job:
                raise ValueError('Checkpoint identity differs from its planned run')
            if record.get('status') == 'complete':
                done += 1
                counts['completed'] += 1
                name = record.get('completion_class')
                classes[name if name in classes else 'other_completed'] += 1
                if name == 'local_limit_before_B':
                    counts['censored'] += 1
                else:
                    eligible.add((budget, job['board_id'], job['method']))
                if name == 'solved':
                    counts['solved'] += 1
                    solved.add((budget, job['board_id'], job['method']))
            elif record.get('status') == 'failed':
                failed += 1
                counts['failed'] += 1
            else:
                raise ValueError('Unknown checkpoint status')
        elif started.exists():
            if job['key'] in active:
                in_flight += 1
                counts['in_flight'] += 1
            else:
                failed += 1
                counts['failed'] += 1
        else:
            counts['missing'] += 1
    for budget, group in per_budget.items():
        board_ids = {job['board_id'] for job in jobs if job['budget'] == budget}
        paired = {board for board in board_ids if all((budget, board, method) in eligible for method in METHODS)}
        group['paired_uncensored_boards'] = len(paired)
        for method in METHODS:
            group['methods'][method]['paired_solved'] = sum((budget, board, method) in solved for board in paired)
    return {'expected': len(jobs), 'completed': done, 'failed': failed, 'in_flight': in_flight,
            'missing': len(jobs) - done - failed - in_flight,
            **classes, 'by_budget': per_budget}


def write_progress(output: Path, state: dict[str, Any]) -> None:
    """Export durable aggregate CSV with explicit censoring and paired denominators."""
    write_json(output / 'progress.json', state)
    fields = ('family', 'budget', 'method', 'expected', 'completed', 'failed', 'in_flight', 'missing', 'censored',
              'solved', 'paired_uncensored_boards', 'paired_solved', 'paired_solve_rate')
    with NamedTemporaryFile('w', encoding='utf-8', newline='', dir=output, prefix='.summary.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for budget, group in state['by_budget'].items():
                for method, counts in group['methods'].items():
                    paired = group['paired_uncensored_boards']
                    writer.writerow({'family': group['family'], 'budget': budget, 'method': method, **counts,
                                     'paired_uncensored_boards': paired,
                                     'paired_solve_rate': counts['paired_solved'] / paired if paired else ''})
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(output / 'summary.csv')
    finally:
        temporary.unlink(missing_ok=True)


def completion_class(outcome: SearchResult, budget: int, *, failed: bool) -> str:
    """Separate a local-limit-censored trial from a solved or fully budgeted run."""
    if failed:
        return 'failed'
    if outcome.solved:
        return 'solved'
    if outcome.finish_reason in {'api_budget', 'expansion_budget'} and outcome.attempted_actions < budget:
        return 'local_limit_before_B'
    return outcome.finish_reason


def create_pilot_api(config: ExperimentConfig, freeze: dict[str, Any], method: str,
                     signals: StopSignals) -> TrackingAPI:
    """Bind an independent client to its stage without changing process-global environment."""
    api = TrackingAPI(config.path('cache_dir'), 'openai', config.limits,
                      max_generator_calls=freeze['max_generator_calls'], max_total_tokens=freeze['max_total_tokens'],
                      method=method, signals=signals)
    api.stage = STAGES[method]
    return api


def execute_one(job: dict[str, Any], freeze: dict[str, Any], config: ExperimentConfig,
                training: Any, item: Any, trace_path: Path, signals: StopSignals) -> dict[str, Any]:
    """Execute a single trial with a durable real-action trace and campaign guard."""
    resolved = ExperimentConfig.model_validate({**config.model_dump(), 'action_budget': job['budget']})
    reference = components_for_run(resolved, 'replay', training, from_rows(item['state']))
    api = create_pilot_api(resolved, freeze, job['method'], signals)
    api.budget_tag = f"small_pilot:{job['key']}"
    from carl.icl.luna import LunaComponents  # pylint: disable=import-outside-toplevel
    tree = DurableTree(trace_path)
    env = DurableEnvironment(environment(from_rows(item['state'])), tree)
    component = LunaComponents(resolved, api, env, training, calibrated=reference.calibrated)
    unexpected = None
    try:
        outcome = search(env, component, from_rows(item['state']), resolved, job['seed'], observer=tree)
    except BaseException as error:  # Retain partial trace; never silently repeat a paid attempt.
        unexpected = type(error).__name__
        if isinstance(error, KeyboardInterrupt):
            mark_interrupted_unknown(job['method'], api)
        outcome = tree.result if tree.result is not None else SearchResult()
        outcome.finish_reason, outcome.error = 'api_error', f'Unexpected search exception: {unexpected}'
        if trace_path.exists():
            outcome.attempted_actions = sum(json.loads(line).get('event') == 'action'
                                            for line in trace_path.read_text().splitlines())
        tree.result = outcome
        tree.append({'event': 'exception', 'error_type': unexpected})
    finally:
        if api.client is not None:
            api.client.close()
    try:
        verification = verify_tree(tree, from_rows(item['state']), job['budget'])
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        verify_actions(tree, from_rows(item['state']), events)
    except Exception as error:  # Persist the failure for inspection, never treat it as completed.
        verification = {'verified': False, 'error_type': type(error).__name__, 'error': str(error)}
    reason = failure_category(outcome)
    budget_scope = (api.cancel_scope or guard_scope(job['method'], api)) if reason == 'campaign_stage_guard' else None
    if api.cancel_scope == 'method_stage' and outcome.finish_reason == 'api_error':
        reason, budget_scope = 'campaign_stage_guard', 'method_stage'
    failed = bool(unexpected or not verification['verified'] or outcome.finish_reason == 'api_error'
                  or reason == 'campaign_stage_guard')
    if failed and reason != 'campaign_stage_guard':
        signals.global_stop.set()
    return {'job': job, 'status': 'failed' if failed else 'complete', 'outcome': asdict(outcome),
            'completion_class': completion_class(outcome, job['budget'], failed=failed),
            'failure_category': reason, 'budget_scope': budget_scope,
            'unexpected_exception_type': unexpected,
            'verification': verification, 'api': dict(api.metrics), 'diagnostics': dict(component.diagnostics),
            'config': resolved.model_dump(mode='json')}


def verify_checkpoint(job: dict[str, Any], record: dict[str, Any], trace_path: Path,
                      config: ExperimentConfig, item: Any) -> None:
    """Replay each completed checkpoint before allowing it to be skipped."""
    resolved = ExperimentConfig.model_validate({**config.model_dump(), 'action_budget': job['budget']})
    if (record.get('job') != job or record.get('status') != 'complete'
            or record.get('config') != resolved.model_dump(mode='json')):
        raise ValueError('Completed trial does not match its frozen identity or configuration')
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    finished = [event for event in events if event['event'] == 'finish']
    if len(finished) != 1 or finished[0]['outcome'] != record['outcome']:
        raise ValueError('Completed trial is missing its matching finish record')
    tree = SearchTreeRecorder()
    tree.nodes = [TreeNode(**event['node']) for event in events if event['event'] in {'start', 'transition'}]
    tree.solution_node = finished[0]['solution_node']
    tree.result = SearchResult(**record['outcome'])
    if (tree.result.finish_reason == 'api_error' or failure_category(tree.result) == 'campaign_stage_guard'
            or record.get('failure_category') != failure_category(tree.result)
            or record.get('completion_class') != completion_class(tree.result, job['budget'], failed=False)):
        raise ValueError('Completed trial contains an error or inconsistent censoring label')
    verification = verify_tree(tree, from_rows(item['state']), job['budget'])
    verify_actions(tree, from_rows(item['state']), events)
    if verification != record['verification']:
        raise ValueError('Completed trial does not reproduce its replay audit')


@contextmanager
def guarded_pool(workers: int, signals: StopSignals, active: dict[Future[dict[str, Any]], dict[str, Any]],
                 trials: Path, jobs: list[dict[str, Any]], output: Path) -> Iterator[ThreadPoolExecutor]:
    """Stop new calls and retain all settled in-flight outcomes on any outer exception."""
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        yield pool
    except BaseException:
        signals.global_stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        for future, job in active.items():
            if future.cancelled():
                continue
            try:
                result = future.result()
                if result.get('job') == job and result.get('status') in {'complete', 'failed'}:
                    write_json(trials / f"{job['key']}.json", result)
            except BaseException:
                # Preserve started marker and action trace for manual inspection.
                continue
        try:
            write_progress(output, summarize_jobs(jobs, output))
        except Exception:
            pass  # Keep the original exception and the individual checkpoints.
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def run_jobs(jobs: list[dict[str, Any]], output: Path, executor: Callable[[dict[str, Any], Path], dict[str, Any]],
             *, resume: bool, validator: Callable[[dict[str, Any], dict[str, Any], Path], None] | None = None,
             initially_blocked: set[str] | None = None, workers: int = 1,
             signals: StopSignals | None = None) -> dict[str, Any]:
    """Run bounded parallel cases without crossing an incomplete budget phase."""
    if not 1 <= workers <= 3:
        raise ValueError('Pilot worker count must be 1, 2 or 3')
    signals = signals or StopSignals()
    trials = output / 'trials'
    trials.mkdir(parents=True, exist_ok=True)
    prior = summarize_jobs(jobs, output)
    if (prior['completed'] or prior['failed']) and not resume:
        raise ValueError('Existing pilot trials require --resume')
    blocked = set(initially_blocked or ())
    for job in jobs:
        path = trials / f"{job['key']}.json"
        started = trials / f"{job['key']}.started.json"
        if started.exists() and not path.exists():
            raise ValueError('Interrupted pilot trial requires inspection; no automatic retry')
        if path.exists():
            record = json.loads(path.read_text())
            if record.get('status') == 'failed':
                if record.get('failure_category') == 'campaign_stage_guard' and record.get('budget_scope') == 'method_stage':
                    blocked.add(job['method'])
                else:
                    raise ValueError('Failed pilot trial requires inspection; no automatic retry')
    budgets = list(dict.fromkeys(job['budget'] for job in jobs))
    active: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with guarded_pool(workers, signals, active, trials, jobs, output) as pool:
        for budget in budgets:
            phase = [job for job in jobs if job['budget'] == budget]
            cursor = 0
            active.clear()
            stop_scheduling = False
            interruption: BaseException | None = None
            while cursor < len(phase) or active:
                while not stop_scheduling and not signals.global_stop.is_set() and cursor < len(phase) and len(active) < workers:
                    job = phase[cursor]
                    cursor += 1
                    record_path = trials / f"{job['key']}.json"
                    started = trials / f"{job['key']}.started.json"
                    trace = trials / f"{job['key']}.trace.jsonl"
                    if record_path.exists():
                        record = json.loads(record_path.read_text())
                        if validator is not None and record['status'] == 'complete':
                            validator(job, record, trace)
                        continue
                    if job['method'] in blocked or signals.method_stop[job['method']].is_set():
                        continue
                    if started.exists() or trace.exists():
                        raise ValueError('Unfinished trial exists; inspect before any retry')
                    write_json(started, {'job': job, 'status': 'started'})
                    active[pool.submit(executor, job, trace)] = job
                if not active:
                    break
                try:
                    finished, _ = wait(active, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt as error:
                    # No new requests after the next API guard; allow already
                    # in-flight responses to settle before returning control.
                    interruption = error
                    signals.global_stop.set()
                    stop_scheduling = True
                    continue
                for future in finished:
                    job = active[future]
                    try:
                        result = future.result()
                    except BaseException as error:
                        active.pop(future)
                        interruption = error
                        signals.global_stop.set()
                        stop_scheduling = True
                        continue  # Started/trace remain explicit incomplete evidence.
                    if result.get('job') != job or result.get('status') not in {'complete', 'failed'}:
                        active.pop(future)
                        signals.global_stop.set()
                        stop_scheduling = True
                        interruption = ValueError('Executor did not return a properly keyed result')
                        continue
                    write_json(trials / f"{job['key']}.json", result)
                    active.pop(future)
                    if result['status'] == 'failed':
                        if result.get('failure_category') == 'campaign_stage_guard' and result.get('budget_scope') == 'method_stage':
                            blocked.add(job['method'])
                            signals.method_stop[job['method']].set()
                        else:
                            signals.global_stop.set()
                            stop_scheduling = True
                write_progress(output, summarize_jobs(jobs, output, {job['key'] for job in active.values()}))
            phase_status = summarize_jobs(phase, output)
            if interruption is not None:
                write_progress(output, summarize_jobs(jobs, output))
                raise interruption
            if signals.global_stop.is_set() or phase_status['completed'] != len(phase) or phase_status['local_limit_before_B']:
                break
    state = summarize_jobs(jobs, output)
    write_progress(output, state)
    return state


def main(freeze_path: str, output_path: str | None, *, plan_only: bool, resume: bool, workers: int = 2) -> int:
    """Validate freeze, print the full plan, then optionally execute paid trials."""
    freeze, freeze_hash, configs, training, items = validate_freeze(Path(freeze_path).expanduser().resolve())
    jobs = schedule(freeze, freeze_hash)
    output = Path(output_path).expanduser() if output_path else ROOT / 'analysis/luna_icl/campaign_100/pilot_small_v1'
    output = (output if output.is_absolute() else ROOT / output).resolve()
    pricing = json.loads((ROOT / 'configs/icl/campaign_budget.json').read_text())
    ledger = ledger_snapshot(ROOT / pricing['ledger_path'], pricing)
    progress = summarize_jobs(jobs, output)
    by_id = {row['id']: row for row in items}
    for job in jobs:
        case_path = output / 'trials' / f"{job['key']}.json"
        if case_path.exists():
            record = json.loads(case_path.read_text())
            if record.get('status') == 'complete':
                verify_checkpoint(job, record, output / 'trials' / f"{job['key']}.trace.jsonl",
                                  configs[job['method']], by_id[job['board_id']])
    plan = {'freeze_hash': freeze_hash, 'family': freeze['family'], 'seed': freeze['seed'],
            'workers': workers,
            'board_ids': [board['id'] for board in freeze['boards']], 'methods': list(METHODS),
            'scales_by_method': {method: configs[method].scales for method in METHODS},
            'budgets': freeze['budgets'], 'main_budget': freeze['main_budget'],
            'budget_execution_order': [freeze['main_budget'], *(b for b in freeze['budgets'] if b != freeze['main_budget'])],
            'num_subgoals': configs[METHODS[0]].num_subgoals,
            'max_generator_calls': freeze['max_generator_calls'], 'max_total_tokens': freeze['max_total_tokens'],
            'http_retry_policy': HTTP_RETRY_POLICY,
            'max_expansions': configs[METHODS[0]].max_expansions,
            'request_token_time_limits': configs[METHODS[0]].limits.model_dump(mode='json'),
            'validation_cost_projection': freeze.get('cost_projection'),
            'max_cost_and_remaining_limits': cost_plan(freeze, configs, ledger, pricing), 'progress': progress}
    print(json.dumps(plan, indent=2), flush=True)
    if plan_only:
        return 0
    if ledger['remaining_usd'] <= 0:
        raise ValueError('Campaign has no remaining budget')
    credentials('openai')
    with _sweep_lock(output):
        saved = output / 'pilot_manifest.json'
        manifest = {'freeze_hash': freeze_hash, 'provider': 'openai', 'expected': len(jobs), 'workers': workers}
        if saved.exists() and json.loads(saved.read_text()) != manifest:
            raise ValueError('Existing pilot output belongs to a different freeze')
        if not saved.exists():
            if resume:
                raise ValueError('Cannot resume without the original pilot manifest')
            if any(path.name != '.sweep.lock' for path in output.iterdir()):
                raise ValueError('Output is nonempty and has no matching pilot manifest')
            write_json(saved, manifest)
        signals = StopSignals()
        final = run_jobs(jobs, output, lambda job, trace: execute_one(job, freeze, configs[job['method']],
                                                                      training, by_id[job['board_id']], trace, signals),
                         resume=resume, validator=lambda job, record, trace: verify_checkpoint(
                             job, record, trace, configs[job['method']], by_id[job['board_id']]),
                         initially_blocked={m for m in METHODS if ledger['stages'][STAGES[m]]['remaining_usd'] <= 0},
                         workers=workers, signals=signals)
        print(json.dumps({'progress': final}, indent=2), flush=True)
        return 1 if final['failed'] or final['missing'] or final['local_limit_before_B'] else 0
