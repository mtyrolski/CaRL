"""Transactional, fail-closed Responses transport for Figure 1.

No credentials are loaded here. The caller must explicitly supply an SDK client,
a Figure1 campaign ledger and the study ledger; this module performs no call at
import and never changes an authorised budget.
"""
import hashlib
import json
import math
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from openai import APITimeoutError

from carl.icl.api import write_json
from carl.icl.campaign_budget import CampaignLimitExceeded
from carl.icl.reachability_data import canonical_bytes
from carl.icl.responses import collect_response
from scripts.figure1_pilot.budget import Figure1Ledger, ReconciliationRequired, _payload_hash
from scripts.reachability_runner.ledger import LimitExceeded, OfflineLedger


class LiveTransportError(RuntimeError):
    """A request remains unresolved; details contain no provider exception text."""

    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


def _upper_usd(campaign: Figure1Ledger, payload: dict, *, count_id: str | None = None) -> str:
    """Same conservative quote as CampaignLedger.reserve, in USD."""
    return str(Decimal(campaign.quote_reservation('responses', payload, count_id=count_id)) / Decimal(1_000_000_000))


def _dump(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, 'model_dump', None)
    if not callable(dump):
        raise ValueError('Provider value is not serializable')
    value = dump(mode='json', exclude_unset=True)
    if not isinstance(value, dict):
        raise ValueError('Provider value is not an object')
    return value


def _charge(campaign: Figure1Ledger, identifier: str) -> str:
    with campaign._transaction() as db:
        row = db.execute('SELECT charged_nanos,status FROM attempts WHERE id=?', (identifier,)).fetchone()
    if row is None or row[1] != 'settled' or row[0] is None:
        raise ValueError('Campaign settlement cannot be verified')
    return str(Decimal(row[0]) / Decimal(1_000_000_000))


def response_digest(cached: dict) -> str:
    """Digest the returned text, resolved model, usage and raw provider response."""
    return hashlib.sha256(canonical_bytes({key: cached.get(key) for key in
        ('requested_model', 'model', 'output_text', 'usage', 'response')})).hexdigest()


def _cached_campaign_matches(campaign: Figure1Ledger, cached: dict, payload: dict, digest: str) -> bool:
    """Do not consume a vector until its physical call is settled in this campaign."""
    identifier = cached.get('campaign_reservation_id')
    if not isinstance(identifier, str) or not identifier:
        return False
    with campaign._transaction() as db:
        row = db.execute('SELECT a.model,a.request_hash,a.status,a.usage_json,a.charged_nanos,c.payload_sha256,c.response_sha256 '
                         'FROM attempts a JOIN figure1_contracts c ON c.attempt_id=a.id WHERE a.id=?',
                         (identifier,)).fetchone()
    if row is None:
        return False
    try:
        matched = (row[0] == payload['model'] and row[1] == digest and row[2] == 'settled' and row[4] is not None
                and json.loads(row[3]) == cached.get('usage') and row[5] == _payload_hash(payload)
                and cached.get('requested_model') == payload['model']
                and isinstance(cached.get('response'), dict)
                and cached['response'].get('model') == cached.get('model')
                and row[6] == response_digest(cached))
        if not matched or campaign.count_options is None:
            return matched
        with campaign._transaction() as db:
            count = db.execute('SELECT c.attempt_id,c.input_tokens,a.status FROM figure1_generation_count_links l '
                               'JOIN figure1_input_counts c ON c.attempt_id=l.count_attempt_id '
                               'JOIN attempts a ON a.id=c.attempt_id WHERE l.attempt_id=?', (identifier,)).fetchone()
        return (count is not None and count[0] == cached.get('count_reservation_id')
                and count[1] == cached.get('usage', {}).get('input_tokens')
                and count[2] == 'accounted_assumption' and cached['response'].get('service_tier') == 'default')
    except (TypeError, ValueError):
        return False


def call_responses(*, request: dict, cache_file: Path, study_ledger: OfflineLedger,
                   campaign_ledger: Figure1Ledger, job: dict, attempt: int,
                   client: Any, allowed_response_models: tuple[str, ...], timeout_seconds: float = 60) -> tuple[dict, dict]:
    """Get/create one response, preserving uncertain reservations and cache errors.

    A second process cannot issue the same request while the cache lock is held.
    An entry is published as pending before either reservation and before any
    provider call. A crash at any later point therefore fails closed on restart.
    """
    if not isinstance(campaign_ledger, Figure1Ledger) or not isinstance(study_ledger, OfflineLedger):
        raise ValueError('Both Figure 1 and study ledgers are mandatory')
    if getattr(client, 'max_retries', None) != 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('SDK retries must be disabled and timeout must be positive')
    if not allowed_response_models or any(not isinstance(model, str) or not model for model in allowed_response_models):
        raise ValueError('Explicit resolved-model allowlist is required')
    if request.get('endpoint') != 'responses' or not isinstance(request.get('payload'), dict):
        raise ValueError('Expected a Responses request')
    payload = request['payload']
    if (not isinstance(request.get('slot'), int) or isinstance(request.get('slot'), bool) or request['slot'] < 0
            or not isinstance(payload.get('model'), str) or not request.get('profile_sha256')):
        raise ValueError('A model and retry slot are required')
    if job.get('model') != payload['model']:
        raise ValueError('Request model differs from the frozen trial identity')
    digest = hashlib.sha256(canonical_bytes(request)).hexdigest()
    if cache_file.stem != digest:
        raise ValueError('Cache filename must be the complete request digest')
    stage, arm, slot = job['group'], job['arm'], request['slot']
    tag = f"{job['key']}:{attempt}:{slot}"
    def cache_error(reason):
        logical_id = study_ledger.reserve(job=job['key'], attempt=attempt, stage=stage, arm=arm,
                                          request_hash=digest, slot=slot, cache_hit=True, reserve_usd='0')
        study_ledger.finish(logical_id, status='cache_error', charged_usd='0', usage={}, reason=reason)
        campaign_ledger.event(stage, reason, 'responses', digest, tag)
        raise LiveTransportError(reason)
    campaign_ledger.event(stage, 'logical_request', 'responses', digest, tag)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(cache_file) + '.lock', timeout=timeout_seconds):
            existing = None
            try:
                if cache_file.exists():
                    existing = json.loads(cache_file.read_text(encoding='utf-8'))
                    if not isinstance(existing, dict) or existing.get('request') != request:
                        raise ValueError('Corrupt or mismatched cache')
            except (OSError, UnicodeError, ValueError):
                cache_error('corrupt_cache_requires_reconciliation')
            if existing is not None:
                if (existing.get('status') != 'completed' or not isinstance(existing.get('output_text'), str)
                        or not isinstance(existing.get('model'), str) or not isinstance(existing.get('usage'), dict)):
                    cache_error('cached_error_requires_reconciliation')
                if existing['model'] not in allowed_response_models:
                    cache_error('unexpected_resolved_model')
                if not _cached_campaign_matches(campaign_ledger, existing, payload, digest):
                    cache_error('cached_campaign_mismatch_requires_reconciliation')
                logical_id = study_ledger.reserve(job=job['key'], attempt=attempt, stage=stage, arm=arm,
                                                  request_hash=digest, slot=slot, cache_hit=True, reserve_usd='0')
                study_ledger.finish(logical_id, status='cache_hit', charged_usd='0', usage=existing['usage'])
                campaign_ledger.event(stage, 'local_cache_hit', 'responses', digest, tag)
                return existing, {'request_hash': digest, 'slot': slot, 'cache_hit': True, 'ledger_id': logical_id,
                                  'campaign_reservation_id': existing.get('campaign_reservation_id')}
            pending = {'request': request, 'status': 'pending', 'provider': 'openai', 'reason': 'not_yet_resolved'}
            write_json(cache_file, pending)
            study_id = None
            campaign_id = None
            count_id = None
            count_value = None
            try:
                if campaign_ledger.count_options is not None:
                    from scripts.figure1_pilot.exact_token_preflight import project_count_request
                    count_payload = project_count_request(payload)
                    count_id = campaign_ledger.reserve_input_count(stage, payload, count_payload, digest, tag)
                    pending['count_reservation_id'] = count_id
                    write_json(cache_file, pending)
                    count_raw = None
                    try:
                        counted = client.responses.input_tokens.count(**count_payload, timeout=timeout_seconds)
                        count_raw = _dump(counted)
                        count_value = campaign_ledger.finish_input_count(count_id, payload, count_payload, count_raw)
                    except Exception:
                        campaign_ledger.mark_unknown(count_id)
                        pending.update(status='error', reason='token_count_unknown', count_reservation_id=count_id,
                                       count_response=count_raw)
                        write_json(cache_file, pending)
                        campaign_ledger.event(stage, 'token_count_unknown', 'responses.input_tokens', digest, tag)
                        raise LiveTransportError('token_count_unknown') from None
                    pending['provider_input_tokens'] = count_value
                    write_json(cache_file, pending)
                    campaign_ledger.event(stage, 'token_count_accounted_assumption', 'responses.input_tokens', digest, tag)
                token_reservation = 0
                if study_ledger.limits.get('token_limit') is not None:
                    token_reservation = (int(count_value if count_value is not None else
                                             campaign_ledger.config['models'][payload['model']]['max_input_tokens'])
                                         + int(payload['max_output_tokens']))
                study_id = study_ledger.reserve(job=job['key'], attempt=attempt, stage=stage, arm=arm,
                                                 request_hash=digest, slot=slot, cache_hit=False,
                                                 reserve_usd=_upper_usd(campaign_ledger, payload, count_id=count_id),
                                                 reserve_tokens=token_reservation)
                pending['study_ledger_id'] = study_id
                write_json(cache_file, pending)
                campaign_id = campaign_ledger.reserve(stage, 'responses', payload, digest, tag, count_id=count_id)
                pending['campaign_reservation_id'] = campaign_id
                write_json(cache_file, pending)
            except (CampaignLimitExceeded, LimitExceeded) as exc:
                reason = 'reconciliation_required' if isinstance(exc, ReconciliationRequired) else 'budget_limit'
                pending.update(status='blocked_before_send', reason=reason)
                write_json(cache_file, pending)
                if study_id is not None:
                    study_ledger.finish(study_id, status='cache_error', charged_usd='0', usage={}, reason=reason)
                raise LiveTransportError(reason) from exc
            # Both durable reservations and the pending record exist before this point.
            started = time.perf_counter()
            captured: dict[str, Any] = {}
            settled = False
            study_finished = False
            try:
                def capture(terminal):
                    captured.update(usage=_dump(getattr(terminal, 'usage', None)),
                                    model=getattr(terminal, 'model', None), response=_dump(terminal))
                response = client.responses.create(**payload, timeout=timeout_seconds)
                if payload.get('stream'):
                    response = collect_response(response, deadline=started + timeout_seconds, on_terminal=capture)
                capture(response)
                raw = captured['response']
                usage = captured['usage']
                if count_id is not None and raw.get('service_tier') != 'default':
                    raise LiveTransportError('unexpected_or_missing_service_tier')
                text = getattr(response, 'output_text', None)
                refused = any(getattr(part, 'type', None) == 'refusal'
                              for item in getattr(response, 'output', []) if getattr(item, 'type', None) == 'message'
                              for part in getattr(item, 'content', []))
                if getattr(response, 'status', None) != 'completed' or refused or not isinstance(text, str) or not text:
                    raise LiveTransportError('provider_incomplete_or_refused')
                if not isinstance(captured['model'], str):
                    raise LiveTransportError('provider_model_missing')
                if captured['model'] not in allowed_response_models:
                    raise LiveTransportError('unexpected_resolved_model')
                settled = campaign_ledger.settle(campaign_id, usage, payload)
                if not settled:
                    raise LiveTransportError('billing_usage_unknown')
                study_ledger.finish(study_id, status='settled', charged_usd=_charge(campaign_ledger, campaign_id), usage=usage)
                study_finished = True
                completed = {'request': request, 'status': 'completed', 'provider': 'openai',
                             'requested_model': payload['model'], 'model': captured['model'], 'response': raw,
                             'response_id': getattr(response, 'id', None), 'output_text': text, 'usage': usage,
                             'seconds': time.perf_counter() - started, 'campaign_reservation_id': campaign_id}
                if count_id is not None:
                    completed.update(count_reservation_id=count_id, provider_input_tokens=count_value)
                campaign_ledger.bind_response(campaign_id, response_digest(completed))
                write_json(cache_file, completed)
                campaign_ledger.event(stage, 'physical_completed', 'responses', digest, tag)
                return completed, {'request_hash': digest, 'slot': slot, 'cache_hit': False, 'ledger_id': study_id,
                                   'campaign_reservation_id': campaign_id}
            except Exception as exc:
                reason = exc.reason_code if isinstance(exc, LiveTransportError) else (
                    'timeout_billing_unknown' if isinstance(exc, (TimeoutError, APITimeoutError)) else 'transport_or_provider_error')
                usage = captured.get('usage', {})
                if (not settled and captured.get('model') in allowed_response_models
                        and reason != 'unexpected_or_missing_service_tier'):
                    try:
                        settled = campaign_ledger.settle(campaign_id, usage, payload)
                    except Exception:
                        settled = False
                if not settled:
                    campaign_ledger.mark_unknown(campaign_id)
                if not study_finished:
                    if settled:
                        study_ledger.finish(study_id, status='settled', charged_usd=_charge(campaign_ledger, campaign_id),
                                            usage=usage, reason=reason)
                    else:
                        study_ledger.finish(study_id, status='unknown', charged_usd=None, usage=usage, reason=reason)
                pending.update(status='error', reason=reason, usage=usage, model=captured.get('model'),
                               response=captured.get('response'), seconds=time.perf_counter() - started)
                write_json(cache_file, pending)
                campaign_ledger.event(stage, 'physical_error', 'responses', digest, tag)
                raise LiveTransportError(reason) from None
    except Timeout:
        cache_error('cache_lock_timeout')
