"""Transactional, cross-process budget reservations for inference campaigns."""
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TypeGuard
from uuid import uuid4

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/icl/campaign_budget.json"
NANOS = Decimal(1_000_000_000)
MILLION = Decimal(1_000_000)
SUPPORTED_ROUTES = {
    ("openai", "responses"),
    ("openai", "embeddings"),
    ("mistral", "chat_completions"),
}


class CampaignLimitExceeded(RuntimeError):
    """A shared campaign or stage spending ceiling would be exceeded."""


def _nanos(value: Decimal) -> int:
    return int((value * NANOS).to_integral_value(rounding=ROUND_CEILING))


def _rate(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("Campaign prices and budgets must be finite and nonnegative") from None
    if not result.is_finite() or result < 0:
        raise ValueError("Campaign prices and budgets must be finite and nonnegative")
    return result


def _contains_12h(value: Any) -> bool:
    if isinstance(value, dict):
        return any((key == "ttl" and item == "12h") or _contains_12h(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_12h(item) for item in value)
    return False


def _token_count(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_cap(value: Any) -> int:
    if not _token_count(value) or value == 0:
        raise ValueError("Campaign price context and output caps must be positive integers")
    return value


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class _Price:
    ordinary: Decimal
    cached: Decimal
    write30: Decimal
    write12: Decimal
    output: Decimal
    max_input: int
    max_output: int
    long_context: int
    regional: Decimal
    legacy: bool

    @classmethod
    def load(cls, values: dict[str, Any], endpoint: str, *, legacy: bool) -> "_Price":
        max_input = _positive_cap(values["max_input_tokens"])
        ordinary = _rate(values["input_usd_per_million"])
        return cls(
            ordinary=ordinary,
            cached=_rate(values.get("cached_input_usd_per_million", ordinary)),
            write30=_rate(values.get("cache_write_30m_usd_per_million", ordinary)),
            write12=_rate(values.get("cache_write_12h_usd_per_million", ordinary)),
            output=Decimal(0) if endpoint == "embeddings" else _rate(values["output_usd_per_million"]),
            max_input=max_input,
            max_output=0 if endpoint == "embeddings" else _positive_cap(values["max_output_tokens"]),
            long_context=_positive_cap(values.get("long_context_threshold", max_input)),
            regional=_rate(values.get("regional_multiplier", 1)),
            legacy=legacy,
        )

    def multipliers(self, total_in: int) -> tuple[Decimal, Decimal]:
        if total_in > self.long_context:
            return Decimal(2), Decimal("1.5")
        return Decimal(1), Decimal(1)

    def upper_bound(self, endpoint: str, input_cap: int, output_cap: int) -> int:
        if endpoint == "embeddings":
            return _nanos(self.ordinary * input_cap * self.regional / MILLION)
        multipliers = (Decimal(2), Decimal("1.5")) if self.legacy else self.multipliers(input_cap)
        input_multiplier, output_multiplier = multipliers
        highest_input = max(self.ordinary, self.cached, self.write30, self.write12)
        upper = highest_input * input_cap * input_multiplier + self.output * output_cap * output_multiplier
        return _nanos(upper * self.regional / MILLION)


def _text_message(message: Any, *, responses: bool) -> bool:
    roles = {"developer", "system", "user", "assistant"} if responses else {"system", "user", "assistant"}
    if not isinstance(message, dict) or message.get("role") not in roles:
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    return responses and isinstance(content, list) and bool(content) and all(
        isinstance(item, dict)
        and item.get("type") in {"input_text", "output_text"}
        and isinstance(item.get("text"), str)
        for item in content
    )


def _request_caps(endpoint: str, payload: dict[str, Any], price: _Price) -> tuple[int, int]:
    if payload.get("tools") or payload.get("service_tier", "default") != "default":
        raise CampaignLimitExceeded("Campaign permits only default-tier inference without tools")
    if endpoint == "embeddings":
        texts = payload.get("input")
        batch = [texts] if isinstance(texts, str) else texts
        if not isinstance(batch, list) or not batch or any(not isinstance(text, str) or not text for text in batch):
            raise CampaignLimitExceeded("Campaign Embeddings require nonempty text inputs")
        return price.max_input * len(batch), 0

    if any(payload.get(name) is not None for name in ("previous_response_id", "conversation", "prompt")):
        raise CampaignLimitExceeded("Campaign requests must contain their input without server-side prompt state")
    choices = payload.get("n", 1)
    if not _token_count(choices) or choices != 1:
        raise CampaignLimitExceeded("Campaign permits a single choice per request")
    responses = endpoint == "responses"
    maximum = payload.get("max_output_tokens" if responses else "max_tokens")
    if not _token_count(maximum) or not 0 < maximum <= price.max_output:
        raise CampaignLimitExceeded("A supported maximum output token count is required")
    messages = payload.get("input" if responses else "messages")
    if responses and isinstance(messages, str) and messages:
        return price.max_input, maximum
    if not isinstance(messages, list) or not messages or not all(
        _text_message(message, responses=responses) for message in messages
    ):
        raise CampaignLimitExceeded("Campaign requests must contain text messages")
    return price.max_input, maximum


def _input_charge(details: Any, total_in: int, payload: dict[str, Any], price: _Price) -> Decimal | None:
    if details is None:
        details = {}
    if not isinstance(details, dict):
        return None
    cached = details.get("cached_tokens")
    cached = 0 if cached is None else cached
    if not _token_count(cached) or cached > total_in:
        return None
    writes = details.get("cache_write_tokens")
    if writes is None:
        # Missing cache-write detail leaves uncached input potentially chargeable at any configured tier.
        maximum_rate = max(price.ordinary, price.write30, price.write12)
        if price.legacy:
            maximum_rate = max(maximum_rate, price.cached)
            cached = 0
        return cached * price.cached + (total_in - cached) * maximum_rate
    breakdown = details.get("cache_write_tokens_details")
    if not _token_count(writes) or (breakdown is not None and not isinstance(breakdown, dict)):
        return None
    if isinstance(breakdown, dict):
        write30 = breakdown.get("ttl_30m_tokens")
        write12 = breakdown.get("ttl_12h_tokens")
        write30 = 0 if write30 is None else write30
        write12 = 0 if write12 is None else write12
    elif _contains_12h(payload):
        write30, write12 = 0, writes
    else:
        write30, write12 = writes, 0
    if (not _token_count(write30) or not _token_count(write12)
            or write30 + write12 != writes or cached + write30 + write12 > total_in):
        return None
    ordinary = total_in - cached - write30 - write12
    return ordinary * price.ordinary + cached * price.cached + write30 * price.write30 + write12 * price.write12


class CampaignLedger:
    """Reserve against immutable prices keyed by provider/endpoint/model.

    Legacy ``models`` pricing remains scoped to OpenAI Responses. Route
    input caps refer to the provider's maximum, per text for Embeddings.
    """

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        raw = config_path.read_bytes()
        self.config = json.loads(raw)
        self.config_hash = hashlib.sha256(raw).hexdigest()
        self.limit = _nanos(_rate(self.config["limit_usd"]))
        external_limit = os.environ.get("CARL_CAMPAIGN_MAX_USD")
        dotenv_limit = dotenv_values(ROOT / ".env").get("CARL_CAMPAIGN_MAX_USD")
        requested = [self.limit]
        requested.extend(_nanos(_rate(value)) for value in (external_limit, dotenv_limit) if value is not None)
        self.requested_limit = min(requested)
        self.stages = {name: _nanos(_rate(value)) for name, value in self.config["stage_limits_usd"].items()}
        if sum(self.stages.values()) > self.limit:
            raise ValueError("Campaign stage allocations exceed campaign budget")
        path = Path(self.config["ledger_path"]).expanduser()
        self.path = path if path.is_absolute() else ROOT / path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, stage TEXT NOT NULL, tag TEXT NOT NULL, "
                       "kind TEXT NOT NULL, endpoint TEXT NOT NULL, request_hash TEXT NOT NULL, "
                       "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            db.execute("CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, stage TEXT NOT NULL, tag TEXT NOT NULL, "
                       "endpoint TEXT NOT NULL, model TEXT NOT NULL, request_hash TEXT NOT NULL, "
                       "reserved_nanos INTEGER NOT NULL, charged_nanos INTEGER, status TEXT NOT NULL, "
                       "usage_json TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                       "provider TEXT NOT NULL DEFAULT 'openai', payload_hash TEXT)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
            if "provider" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN provider TEXT NOT NULL DEFAULT 'openai'")
            if "payload_hash" not in columns:
                db.execute("ALTER TABLE attempts ADD COLUMN payload_hash TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS attempts_request ON attempts(provider,endpoint,request_hash,status)")
            db.execute("INSERT OR IGNORE INTO metadata (key,value) VALUES ('config_hash',?)", (self.config_hash,))
            db.execute("INSERT OR IGNORE INTO metadata (key,value) VALUES ('effective_limit_nanos',?)",
                       (str(self.requested_limit),))
            db.execute("UPDATE metadata SET value=? WHERE key='effective_limit_nanos' AND CAST(value AS INTEGER)>?",
                       (str(self.requested_limit), self.requested_limit))
            saved = db.execute("SELECT value FROM metadata WHERE key='config_hash'").fetchone()
            if saved is None or saved[0] != self.config_hash:
                raise ValueError("Budget configuration changed for this ledger; preserve and reconcile it before use")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def _stage(self, stage: str) -> None:
        if stage not in self.stages:
            raise ValueError("Unknown campaign stage")

    # Preserve the public positional tag argument used by existing ledger adapters.
    # pylint: disable-next=too-many-positional-arguments
    def event(self, stage: str, kind: str, endpoint: str, digest: str, tag: str = "") -> None:
        """Persist logical requests and zero-cost local cache hits separately."""
        self._stage(stage)
        with self._transaction() as db:
            db.execute("INSERT INTO events(stage,tag,kind,endpoint,request_hash) VALUES (?,?,?,?,?)",
                       (stage, tag, kind, endpoint, digest))

    def _price(self, provider: str, endpoint: str, model: str) -> _Price:
        if (provider, endpoint) not in SUPPORTED_ROUTES:
            raise CampaignLimitExceeded("Unsupported campaign billing route")
        routes = self.config.get("routes", {})
        if not isinstance(routes, dict):
            raise CampaignLimitExceeded("Campaign route prices must be an object")
        key = f"{provider}/{endpoint}/{model}"
        values = routes.get(key)
        legacy = key not in routes and (provider, endpoint) == ("openai", "responses")
        if legacy:
            models = self.config.get("models", {})
            if isinstance(models, dict):
                values = models.get(model, models.get("*"))
        if not isinstance(values, dict):
            raise CampaignLimitExceeded("No configured campaign pricing for this route and model")
        try:
            return _Price.load(values, endpoint, legacy=legacy)
        except (KeyError, ValueError):
            raise CampaignLimitExceeded("Invalid campaign price for this route and model") from None

    # Preserve the public positional tag argument used by existing ledger adapters.
    # pylint: disable-next=too-many-positional-arguments
    def reserve(
        self, stage: str, endpoint: str, payload: dict[str, Any], digest: str, tag: str = "",
        *, provider: str = "openai", protect_request: bool = False,
    ) -> str:
        """Reserve worst-case charge atomically before a physical API attempt."""
        self._stage(stage)
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise CampaignLimitExceeded("A priced model is required before a physical request")
        price = self._price(provider, endpoint, model)
        input_cap, output_cap = _request_caps(endpoint, payload, price)
        reserved = price.upper_bound(endpoint, input_cap, output_cap)
        wire_hash = _payload_hash(payload)
        identifier = uuid4().hex
        with self._transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            if protect_request and db.execute(
                "SELECT 1 FROM attempts WHERE provider=? AND endpoint=? AND request_hash=? "
                "AND status IN ('reserved','unknown') LIMIT 1", (provider, endpoint, digest)
            ).fetchone():
                raise CampaignLimitExceeded("An earlier physical attempt requires reconciliation")
            charged = "COALESCE(charged_nanos, reserved_nanos)"
            overall = int(db.execute(f"SELECT COALESCE(SUM({charged}),0) FROM attempts").fetchone()[0])
            subtotal = int(db.execute(f"SELECT COALESCE(SUM({charged}),0) FROM attempts WHERE stage=?", (stage,)).fetchone()[0])
            effective = int(db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()[0])
            if overall + reserved > effective or subtotal + reserved > self.stages[stage]:
                raise CampaignLimitExceeded("Campaign or stage budget would be exceeded")
            db.execute("INSERT INTO attempts(id,stage,tag,endpoint,model,request_hash,reserved_nanos,status,provider,payload_hash) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (identifier, stage, tag, endpoint, model, digest, reserved, "reserved", provider, wire_hash))
        return identifier

    def settle(self, identifier: str, usage: dict[str, Any], payload: dict[str, Any]) -> bool:
        """Settle known token usage; leave reservations untouched when usage is unknown."""
        total_in, output = usage.get("input_tokens"), usage.get("output_tokens")
        if not _token_count(total_in) or total_in == 0 or not _token_count(output):
            return False
        total = usage.get("total_tokens")
        if total is not None and (not _token_count(total) or total != total_in + output):
            return False
        with self._transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            attempt = db.execute(
                "SELECT reserved_nanos,status,provider,endpoint,model,payload_hash FROM attempts WHERE id=?",
                (identifier,),
            ).fetchone()
            if attempt is None or attempt[1] != "reserved":
                raise ValueError("Campaign reservation is missing or already settled")
            reserved, _, provider, endpoint, model, wire_hash = attempt
            if payload.get("model") != model or (wire_hash and _payload_hash(payload) != wire_hash):
                return False
            price = self._price(provider, endpoint, model)
            input_cap, output_cap = _request_caps(endpoint, payload, price)
            if total_in > input_cap or output > output_cap:
                return False
            if endpoint == "embeddings":
                cost = total_in * price.ordinary
            else:
                input_cost = _input_charge(usage.get("input_tokens_details"), total_in, payload, price)
                if input_cost is None:
                    return False
                input_multiplier, output_multiplier = price.multipliers(total_in)
                cost = input_cost * input_multiplier + price.output * output * output_multiplier
            charged = _nanos(cost * price.regional / MILLION)
            settled = charged <= reserved
            if settled:
                db.execute("UPDATE attempts SET charged_nanos=?,status='settled',usage_json=? WHERE id=?",
                           (charged, json.dumps(usage, sort_keys=True), identifier))
        return settled

    def mark_unknown(self, identifier: str) -> None:
        """Mark a failed attempt as uncertain without releasing its reservation."""
        with self._transaction() as db:
            db.execute("UPDATE attempts SET status='unknown' WHERE id=? AND status='reserved'", (identifier,))

    def summary(self) -> dict[str, Any]:
        """Return aggregate costs and unsettled reservations, never prompts or credentials."""
        with self._transaction() as db:
            events = db.execute("SELECT stage,tag,kind,COUNT(*) FROM events GROUP BY stage,tag,kind").fetchall()
            attempts = db.execute("SELECT stage,tag,status,COUNT(*),SUM(COALESCE(charged_nanos,reserved_nanos)) "
                                  "FROM attempts GROUP BY stage,tag,status").fetchall()
            effective = int(db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()[0])
        return {"campaign_id": self.config["campaign_id"], "limit_usd": self.limit / 1e9,
                "effective_limit_usd": effective / 1e9,
                "events": [{"stage": s, "tag": tag, "kind": k, "count": n} for s, tag, k, n in events],
                "attempts": [{"stage": s, "tag": tag, "status": status, "count": n, "committed_usd": amount / 1e9}
                             for s, tag, status, n, amount in attempts]}


def campaign_stage() -> str:
    """Require an explicitly assigned cost stage for paid attempts."""
    return os.environ.get("CARL_CAMPAIGN_STAGE", "unassigned")
