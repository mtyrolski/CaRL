"""Fail-closed budget guard for a separately authorised Figure 1 campaign."""
import hashlib
import json
import sqlite3
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from carl.icl.campaign_budget import CampaignLedger, CampaignLimitExceeded, _nanos, _rate

ROOT = Path(__file__).resolve().parents[2]
MILLION = Decimal(1_000_000)


class ReconciliationRequired(CampaignLimitExceeded):
    """An earlier request has unknown billing and blocks new requests."""


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class Figure1Ledger(CampaignLedger):
    """A separate durable campaign with atomic global, stage and model budgets.

    Requires an externally reviewed configuration; a config is not evidence of
    user approval by itself. It does not create a default budget or enable API
    calls. Unknown billing always keeps the original reservation.
    """

    def __init__(self, config_path: Path):
        raw = config_path.read_bytes()
        config = json.loads(raw)
        if not str(config.get('campaign_id', '')).startswith('figure1_'):
            raise ValueError('Figure 1 requires its own explicitly named campaign')
        authorization = config.get('authorization_reference')
        if not isinstance(authorization, str) or not authorization.strip():
            raise ValueError('Explicit budget authorization reference is required')
        if set(config.get('models', {})) != {'gpt-6-luna', 'gpt-6.1-terra'}:
            raise ValueError('Exactly the reviewed Luna and Terra pricing profiles are required')
        if set(config.get('model_limits_usd', {})) != set(config['models']):
            raise ValueError('Both models require explicit cost limits')
        self.count_options = config.get('input_token_reservation')
        if self.count_options is not None:
            options = self.count_options
            if (not isinstance(options, dict) or options.get('mode') != 'provider_exact_v1'
                    or options.get('count_billing_assumption') != 'conservative_model_input_rate_unverified'
                    or not _positive_integer(options.get('max_input_tokens_per_count'))
                    or not _positive_integer(options.get('max_count_calls'))):
                raise ValueError('Invalid opt-in provider input-token reservation configuration')
        for price in config['models'].values():
            if not all(_positive_integer(price.get(key)) for key in ('max_input_tokens', 'max_output_tokens', 'long_context_threshold')):
                raise ValueError('Model token limits and long-context threshold must be positive integers')
            ordinary = _rate(price['input_usd_per_million'])
            write12 = _rate(price['cache_write_12h_usd_per_million'])
            if (ordinary <= 0 or _rate(price['output_usd_per_million']) <= 0 or _rate(price['regional_multiplier']) < 1
                    or write12 < max(ordinary, _rate(price['cached_input_usd_per_million']),
                                     _rate(price['cache_write_30m_usd_per_million']))):
                raise ValueError('Pricing does not support a conservative reservation bound')
        target = Path(config['ledger_path']).expanduser()
        if not target.is_absolute():
            target = ROOT / target
        target = target.resolve()
        if target.is_relative_to((ROOT / 'analysis/luna_icl').resolve()):
            raise ValueError('Cannot use the existing Luna campaign ledger')
        # Never initialise Figure 1 tables in an unrelated pre-existing database.
        if target.exists():
            try:
                with sqlite3.connect(target.as_uri() + '?mode=ro', uri=True) as db:
                    stored = db.execute("SELECT value FROM metadata WHERE key='config_hash'").fetchone()
            except sqlite3.Error as exc:
                raise ValueError('Existing ledger is not a matching Figure 1 ledger') from exc
            if stored is None or stored[0] != hashlib.sha256(raw).hexdigest():
                raise ValueError('Existing ledger is not a matching Figure 1 ledger')
        self.model_limits = {model: _nanos(_rate(amount)) for model, amount in config['model_limits_usd'].items()}
        super().__init__(config_path)
        with self._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS figure1_contracts (attempt_id TEXT PRIMARY KEY, '
                       'payload_sha256 TEXT NOT NULL, response_sha256 TEXT, FOREIGN KEY(attempt_id) REFERENCES attempts(id))')
            if 'response_sha256' not in {row[1] for row in db.execute('PRAGMA table_info(figure1_contracts)')}:
                raise ValueError('Campaign ledger schema predates response integrity; preserve it and use a new campaign')
            if self.count_options is not None:
                db.execute('CREATE TABLE IF NOT EXISTS figure1_input_counts (attempt_id TEXT PRIMARY KEY, '
                           'payload_sha256 TEXT NOT NULL, count_payload_sha256 TEXT NOT NULL, '
                           'payload_json TEXT NOT NULL, count_payload_json TEXT NOT NULL, '
                           'input_tokens INTEGER, raw_response_json TEXT, '
                           'FOREIGN KEY(attempt_id) REFERENCES attempts(id))')
                db.execute('CREATE TABLE IF NOT EXISTS figure1_generation_count_links '
                           '(attempt_id TEXT PRIMARY KEY, count_attempt_id TEXT NOT NULL UNIQUE, '
                           'FOREIGN KEY(attempt_id) REFERENCES attempts(id), '
                           'FOREIGN KEY(count_attempt_id) REFERENCES attempts(id))')
                columns = {row[1] for row in db.execute('PRAGMA table_info(figure1_input_counts)')}
                if not {'payload_json', 'count_payload_json'} <= columns:
                    raise ValueError('Input-count ledger predates integrity checks; preserve and reconcile it')
        if self.count_options is not None:
            self.verify_input_counts()

    def bind_response(self, identifier: str, response_sha256: str) -> None:
        """Bind a settled request to its completed response before cache publication."""
        if len(response_sha256) != 64 or any(c not in '0123456789abcdef' for c in response_sha256):
            raise ValueError('Expected a SHA-256 response digest')
        with self._transaction() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT a.status,c.response_sha256 FROM attempts a '
                             'JOIN figure1_contracts c ON c.attempt_id=a.id WHERE a.id=?', (identifier,)).fetchone()
            if row is None or row[0] != 'settled' or row[1] not in (None, response_sha256):
                raise ValueError('Response cannot be bound to this campaign attempt')
            db.execute('UPDATE figure1_contracts SET response_sha256=? WHERE attempt_id=?', (response_sha256, identifier))

    def quote_reservation(self, endpoint: str, payload: dict[str, Any], *, count_id: str | None = None) -> int:
        """Return the conservative per-request quote as integer nanodollars.

        Validates the request before quoting. This amount is exactly the amount
        inserted by reserve for this endpoint and payload.
        """
        model = payload.get('model')
        if model not in self.model_limits:
            raise CampaignLimitExceeded('Model is not authorized for Figure 1')
        price = self._price(model)
        if endpoint != 'responses' or payload.get('tools', []) or payload.get('service_tier', 'default') != 'default':
            raise CampaignLimitExceeded('Only default-tier text Responses without tools are priced')
        maximum = payload.get('max_output_tokens')
        if not _positive_integer(maximum) or maximum > price['max_output_tokens']:
            raise CampaignLimitExceeded('A supported maximum output token count is required')
        messages = payload.get('input')
        if not isinstance(messages, (str, list)):
            raise CampaignLimitExceeded('Campaign requests must contain text input')
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict) or message.get('role') not in {'developer', 'system', 'user', 'assistant'}:
                raise CampaignLimitExceeded('Only text messages are priced')
            content = message.get('content')
            if isinstance(content, str):
                continue
            if (not isinstance(content, list) or any(not isinstance(part, dict)
                    or part.get('type') not in {'input_text', 'output_text'}
                    or not isinstance(part.get('text'), str) for part in content)):
                raise CampaignLimitExceeded('Only text content is priced')
        _payload_hash(payload)  # Reject NaN and unserialisable requests before reserving.
        input_tokens = price['max_input_tokens']
        if count_id is not None:
            if self.count_options is None:
                raise ValueError('Provider token count mode is not enabled')
            with self._transaction() as db:
                row = db.execute('SELECT a.model,a.status,c.payload_sha256,c.input_tokens FROM attempts a '
                                 'JOIN figure1_input_counts c ON c.attempt_id=a.id WHERE a.id=?', (count_id,)).fetchone()
            if (row is None or row[0] != model or row[1] != 'accounted_assumption'
                    or row[2] != _payload_hash(payload) or not _nonnegative_integer(row[3])
                    or row[3] > self.count_options['max_input_tokens_per_count']):
                raise ValueError('Missing, mismatched or unaccounted provider input-token count')
            input_tokens = row[3]
        upper = (_rate(price['cache_write_12h_usd_per_million']) * 2 * input_tokens
                 + _rate(price['output_usd_per_million']) * Decimal('1.5') * maximum)
        return _nanos(upper * _rate(price['regional_multiplier']) / MILLION)

    def reserve(self, stage: str, endpoint: str, payload: dict[str, Any], digest: str, tag: str = '', *,
                count_id: str | None = None) -> str:
        """Atomically enforce global, stage and per-model caps before inference."""
        if self.count_options is not None:
            self.verify_input_counts()
        self._stage(stage)
        reserved = self.quote_reservation(endpoint, payload, count_id=count_id)
        model = payload['model']
        identifier = uuid4().hex
        with self._transaction() as db:
            db.execute('BEGIN IMMEDIATE')
            if self.count_options is not None and db.execute("SELECT 1 FROM attempts WHERE status='unknown' LIMIT 1").fetchone():
                raise ReconciliationRequired('Unknown campaign billing requires reconciliation')
            committed = 'COALESCE(charged_nanos, reserved_nanos)'
            overall = int(db.execute(f'SELECT COALESCE(SUM({committed}),0) FROM attempts').fetchone()[0])
            subtotal = int(db.execute(f'SELECT COALESCE(SUM({committed}),0) FROM attempts WHERE stage=?', (stage,)).fetchone()[0])
            model_total = int(db.execute(f'SELECT COALESCE(SUM({committed}),0) FROM attempts WHERE model=?', (model,)).fetchone()[0])
            effective = int(db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()[0])
            if overall + reserved > effective or subtotal + reserved > self.stages[stage]:
                raise CampaignLimitExceeded(f'Campaign or stage {stage} budget would be exceeded')
            if model_total + reserved > self.model_limits[model]:
                raise CampaignLimitExceeded('Figure 1 model limit would be exceeded')
            db.execute('INSERT INTO attempts(id,stage,tag,endpoint,model,request_hash,reserved_nanos,status) '
                       'VALUES (?,?,?,?,?,?,?,?)', (identifier, stage, tag, endpoint, model, digest, reserved, 'reserved'))
            db.execute('INSERT INTO figure1_contracts(attempt_id,payload_sha256) VALUES (?,?)',
                       (identifier, _payload_hash(payload)))
            if count_id is not None:
                row = db.execute('SELECT a.request_hash,a.stage FROM attempts a WHERE a.id=?', (count_id,)).fetchone()
                if row != (digest, stage):
                    raise ValueError('Count must belong to this request and stage')
                db.execute('INSERT INTO figure1_generation_count_links(attempt_id,count_attempt_id) VALUES (?,?)',
                           (identifier, count_id))
        return identifier

    def reserve_input_count(self, stage: str, payload: dict[str, Any], count_payload: dict[str, Any],
                            digest: str, tag: str = '') -> str:
        """Record and reserve a counter request before transmission.

        The counter's actual billing is not documented. We reserve under the
        explicit assumption that it costs at most the highest model input rate;
        this is conservative accounting, not a guaranteed invoice bound.
        """
        if self.count_options is None:
            raise ValueError('Provider token count mode is not enabled')
        self.verify_input_counts()
        from scripts.figure1_pilot.exact_token_preflight import project_count_request
        if count_payload != project_count_request(payload):
            raise ValueError('Count payload differs from the request')
        self._stage(stage)
        model = payload['model']
        price = self._price(model)
        limit = self.count_options['max_input_tokens_per_count']
        if limit > price['max_input_tokens']:
            raise ValueError('Count limit exceeds model maximum')
        reserved = _nanos(_rate(price['cache_write_12h_usd_per_million']) * 2 * limit
                          * _rate(price['regional_multiplier']) / MILLION)
        identifier = uuid4().hex
        with self._transaction() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM attempts WHERE status='unknown' LIMIT 1").fetchone():
                raise ReconciliationRequired('Unknown campaign billing requires reconciliation')
            committed = 'COALESCE(charged_nanos,reserved_nanos)'
            overall = int(db.execute(f'SELECT COALESCE(SUM({committed}),0) FROM attempts').fetchone()[0])
            subtotal = int(db.execute(f'SELECT COALESCE(SUM({committed}),0) FROM attempts WHERE stage=?', (stage,)).fetchone()[0])
            model_total = int(db.execute(f'SELECT COALESCE(SUM({committed}),0) FROM attempts WHERE model=?', (model,)).fetchone()[0])
            count = int(db.execute('SELECT COUNT(*) FROM figure1_input_counts').fetchone()[0])
            effective = int(db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()[0])
            if (overall + reserved > effective or subtotal + reserved > self.stages[stage]
                    or model_total + reserved > self.model_limits[model] or count >= self.count_options['max_count_calls']):
                raise CampaignLimitExceeded('Campaign, stage, model or token-count call budget would be exceeded')
            db.execute('INSERT INTO attempts(id,stage,tag,endpoint,model,request_hash,reserved_nanos,status) '
                       'VALUES (?,?,?,?,?,?,?,?)', (identifier, stage, tag, 'responses.input_tokens', model, digest, reserved, 'reserved'))
            db.execute('INSERT INTO figure1_input_counts(attempt_id,payload_sha256,count_payload_sha256,payload_json,count_payload_json) '
                       'VALUES (?,?,?,?,?)', (identifier, _payload_hash(payload), _payload_hash(count_payload),
                       json.dumps(payload, sort_keys=True, allow_nan=False), json.dumps(count_payload, sort_keys=True, allow_nan=False)))
        return identifier

    def finish_input_count(self, identifier: str, payload: dict[str, Any], count_payload: dict[str, Any],
                           raw: dict[str, Any]) -> int:
        """Persist actual count and an explicitly hypothetical counter charge."""
        count = raw.get('input_tokens') if isinstance(raw, dict) else None
        if (not _nonnegative_integer(count) or raw.get('object') != 'response.input_tokens'
                or count > self.count_options['max_input_tokens_per_count']):
            raise ValueError('Provider count missing, malformed or above configured bound')
        price = self._price(payload['model'])
        charged = _nanos(_rate(price['cache_write_12h_usd_per_million']) * 2 * count
                         * _rate(price['regional_multiplier']) / MILLION)
        with self._transaction() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT a.model,a.status,a.reserved_nanos,c.payload_sha256,c.count_payload_sha256 '
                             'FROM attempts a JOIN figure1_input_counts c ON a.id=c.attempt_id WHERE a.id=?', (identifier,)).fetchone()
            if (row is None or row[0] != payload['model'] or row[1] != 'reserved' or charged > row[2]
                    or row[3] != _payload_hash(payload) or row[4] != _payload_hash(count_payload)):
                raise ValueError('Count cannot be bound to reserved request')
            db.execute("UPDATE attempts SET charged_nanos=?,status='accounted_assumption',usage_json=? WHERE id=?",
                       (charged, json.dumps({'input_tokens': count, 'billing': 'unverified_assumption'}, sort_keys=True), identifier))
            db.execute('UPDATE figure1_input_counts SET input_tokens=?,raw_response_json=? WHERE attempt_id=?',
                       (count, json.dumps(raw, sort_keys=True, allow_nan=False), identifier))
        return count

    def verify_input_counts(self) -> dict[str, int]:
        """Recompute the opt-in counter accounting before spending or replay.

        An assumed count charge is not an invoice. Unknown/reserved calls remain
        reserved; their existence is reported rather than treated as a failure.
        """
        if self.count_options is None:
            return {}
        from scripts.figure1_pilot.exact_token_preflight import project_count_request
        statuses: dict[str, int] = {}
        with closing(sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            db.execute('BEGIN')
            counts = db.execute('SELECT a.id,a.stage,a.endpoint,a.model,a.request_hash,a.reserved_nanos,a.charged_nanos,'
                                'a.status,a.usage_json,c.payload_sha256,c.count_payload_sha256,c.payload_json,'
                                'c.count_payload_json,c.input_tokens,c.raw_response_json FROM attempts a '
                                'LEFT JOIN figure1_input_counts c ON c.attempt_id=a.id '
                                "WHERE a.endpoint='responses.input_tokens'").fetchall()
            orphan_counts = db.execute('SELECT COUNT(*) FROM figure1_input_counts c LEFT JOIN attempts a '
                                       "ON a.id=c.attempt_id WHERE a.id IS NULL OR a.endpoint!='responses.input_tokens'").fetchone()[0]
            if orphan_counts:
                raise ValueError('Orphan or misclassified input-token count')
            for row in counts:
                (identifier, _stage, endpoint, model, _digest, reserved, charged, status, usage_json,
                 payload_hash, count_hash, payload_json, count_json, count, raw_json) = row
                try:
                    payload, projection = json.loads(payload_json), json.loads(count_json)
                    price = self._price(model)
                    expected_reserved = _nanos(_rate(price['cache_write_12h_usd_per_million']) * 2
                        * self.count_options['max_input_tokens_per_count'] * _rate(price['regional_multiplier']) / MILLION)
                    if (endpoint != 'responses.input_tokens' or payload.get('model') != model
                            or _payload_hash(payload) != payload_hash or _payload_hash(projection) != count_hash
                            or project_count_request(payload) != projection or reserved != expected_reserved):
                        raise ValueError('Input-token count contract differs from payload or reservation')
                    if status == 'accounted_assumption':
                        raw, usage = json.loads(raw_json), json.loads(usage_json)
                        expected_charged = _nanos(_rate(price['cache_write_12h_usd_per_million']) * 2 * count
                                                  * _rate(price['regional_multiplier']) / MILLION)
                        if (not _nonnegative_integer(count) or count > self.count_options['max_input_tokens_per_count']
                                or raw.get('object') != 'response.input_tokens' or raw.get('input_tokens') != count
                                or usage != {'input_tokens': count, 'billing': 'unverified_assumption'}
                                or charged != expected_charged or charged > reserved):
                            raise ValueError('Input-token count usage or assumed charge differs')
                    elif status in {'reserved', 'unknown'}:
                        if charged is not None or count is not None or raw_json is not None or usage_json is not None:
                            raise ValueError('Unresolved input-token count has inconsistent accounting')
                    else:
                        raise ValueError('Invalid input-token count status')
                except (TypeError, KeyError, json.JSONDecodeError) as exc:
                    raise ValueError('Invalid input-token count record') from exc
                statuses[status] = statuses.get(status, 0) + 1
            links = db.execute('SELECT a.id,a.stage,a.model,a.request_hash,a.status,a.usage_json,l.count_attempt_id,'
                               'c.input_tokens,ca.stage,ca.model,ca.request_hash,ca.status,fc.payload_sha256,c.payload_sha256 '
                               'FROM attempts a LEFT JOIN figure1_generation_count_links l ON l.attempt_id=a.id '
                               'LEFT JOIN figure1_input_counts c ON c.attempt_id=l.count_attempt_id '
                               'LEFT JOIN attempts ca ON ca.id=l.count_attempt_id '
                               'LEFT JOIN figure1_contracts fc ON fc.attempt_id=a.id '
                               "WHERE a.endpoint='responses'").fetchall()
            orphan_links = db.execute('SELECT COUNT(*) FROM figure1_generation_count_links l LEFT JOIN attempts a '
                                      "ON a.id=l.attempt_id WHERE a.id IS NULL OR a.endpoint!='responses'").fetchone()[0]
            if orphan_links:
                raise ValueError('Orphan input-count link')
            for (_id, stage, model, digest, status, usage_json, count_id, count, cstage, cmodel, cdigest,
                 cstatus, generation_hash, count_hash) in links:
                if (not count_id or (stage, model, digest) != (cstage, cmodel, cdigest)
                        or cstatus != 'accounted_assumption' or generation_hash != count_hash):
                    raise ValueError('Generation is not bound to its provider input-token count')
                if status == 'settled':
                    try:
                        if json.loads(usage_json).get('input_tokens') != count:
                            raise ValueError('Generation usage differs from provider count')
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ValueError('Generation usage is invalid') from exc
        return statuses

    @classmethod
    def verify_input_counts_readonly(cls, config: dict[str, Any], path: Path) -> dict[str, int]:
        """Verify a frozen count ledger without creating it or reading .env."""
        guard = cls.__new__(cls)
        guard.config = config
        guard.path = path
        guard.count_options = config.get('input_token_reservation')
        return guard.verify_input_counts()

    def settle(self, identifier: str, usage: dict[str, Any], payload: dict[str, Any]) -> bool:
        """Refuse to settle a reservation against a different or malformed request."""
        with self._transaction() as db:
            row = db.execute('SELECT a.model,c.payload_sha256 FROM attempts a '
                             'JOIN figure1_contracts c ON c.attempt_id=a.id WHERE a.id=?', (identifier,)).fetchone()
        if row is None or row[0] != payload.get('model') or row[1] != _payload_hash(payload):
            raise ValueError('Settlement payload does not match the original reservation')
        if not isinstance(usage, dict) or any(not _nonnegative_integer(usage.get(key)) for key in ('input_tokens', 'output_tokens')):
            return False
        if 'total_tokens' in usage and (not _nonnegative_integer(usage['total_tokens'])
                or usage['total_tokens'] != usage['input_tokens'] + usage['output_tokens']):
            return False
        if self.count_options is not None:
            with self._transaction() as db:
                counted = db.execute('SELECT c.input_tokens FROM figure1_generation_count_links l '
                                     'JOIN figure1_input_counts c ON c.attempt_id=l.count_attempt_id '
                                     'WHERE l.attempt_id=?', (identifier,)).fetchone()
            if counted is None or usage['input_tokens'] != counted[0]:
                return False
        details = usage.get('input_tokens_details')
        if details is not None:
            if not isinstance(details, dict):
                return False
            for key in ('cached_tokens', 'cache_write_tokens'):
                if key in details and details[key] is not None and not _nonnegative_integer(details[key]):
                    return False
            breakdown = details.get('cache_write_tokens_details')
            if breakdown is not None and (not isinstance(breakdown, dict) or any(
                    not _nonnegative_integer(breakdown.get(key, 0)) for key in ('ttl_30m_tokens', 'ttl_12h_tokens'))):
                return False
        return super().settle(identifier, usage, payload)

    def summary(self) -> dict[str, Any]:
        """Distinguish known charged amounts, reserved exposure and usage by model."""
        result = super().summary()
        with self._transaction() as db:
            rows = db.execute('SELECT model,stage,status,COUNT(*),SUM(COALESCE(charged_nanos,0)), '
                              'SUM(CASE WHEN charged_nanos IS NULL THEN reserved_nanos ELSE 0 END) '
                              'FROM attempts GROUP BY model,stage,status ORDER BY model,stage,status').fetchall()
            usage_rows = db.execute("SELECT model,usage_json FROM attempts WHERE status='settled'").fetchall()
            count_rows = db.execute("SELECT status,COUNT(*),SUM(COALESCE(charged_nanos,0)), "
                                    "SUM(CASE WHEN charged_nanos IS NULL THEN reserved_nanos ELSE 0 END) "
                                    "FROM attempts WHERE endpoint='responses.input_tokens' GROUP BY status").fetchall()
        result['by_model_stage_status'] = [{'model': model, 'stage': stage, 'status': status, 'calls': count,
                                             'accounted_cost_usd': charged / 1e9, 'unsettled_reserved_usd': outstanding / 1e9}
                                            for model, stage, status, count, charged, outstanding in rows]
        result['model_limits_usd'] = {model: amount / 1e9 for model, amount in self.model_limits.items()}
        totals: dict[str, dict[str, Any]] = {}
        for model, raw in usage_rows:
            usage = json.loads(raw)
            subtotal = totals.setdefault(model, {'input_tokens': 0, 'output_tokens': 0,
                'provider_cached_input_tokens': 0, 'provider_cache_write_tokens': 0,
                'cached_input_usage_missing': 0, 'cache_write_usage_missing': 0})
            subtotal['input_tokens'] += usage['input_tokens']
            subtotal['output_tokens'] += usage['output_tokens']
            details = usage.get('input_tokens_details') or {}
            for detail, count_key, missing_key in (('cached_tokens', 'provider_cached_input_tokens', 'cached_input_usage_missing'),
                                                    ('cache_write_tokens', 'provider_cache_write_tokens', 'cache_write_usage_missing')):
                if _nonnegative_integer(details.get(detail)):
                    subtotal[count_key] += details[detail]
                else:
                    subtotal[missing_key] += 1
        for subtotal in totals.values():
            if subtotal['cached_input_usage_missing']:
                subtotal['provider_cached_input_tokens'] = None
            if subtotal['cache_write_usage_missing']:
                subtotal['provider_cache_write_tokens'] = None
        result['settled_usage_by_model'] = totals
        result['token_counter_assumptions'] = [{'status': status, 'calls': count,
                                                'assumed_usd': assumed / 1e9,
                                                'unsettled_reserved_usd': pending / 1e9}
                                               for status, count, assumed, pending in count_rows]
        result['cost_note'] = ('Accounted cost uses configured rates and provider usage; when cache-write details are missing '
                               'the inherited guard conservatively prices input as 12-hour cache writes. '
                               'The input-token counter has unverified billing; its separately shown assumed charge '
                               'is not observed provider usage. This is not necessarily the provider invoice.')
        return result
