"""Transactional, cross-process budget reservations for the Luna campaign."""
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/icl/campaign_budget.json"
NANOS = Decimal(1_000_000_000)
MILLION = Decimal(1_000_000)


class CampaignLimitExceeded(RuntimeError):
    """A shared campaign or stage spending ceiling would be exceeded."""


def _nanos(value: Decimal) -> int:
    return int((value * NANOS).to_integral_value(rounding=ROUND_CEILING))


def _rate(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("Campaign prices and budgets must be finite and nonnegative")
    return result


def _contains_12h(value: Any) -> bool:
    if isinstance(value, dict):
        return any((key == "ttl" and item == "12h") or _contains_12h(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_12h(item) for item in value)
    return False


class CampaignLedger:
    """One immutable price configuration and durable reservations shared by all clients."""

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
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, stage TEXT NOT NULL, tag TEXT NOT NULL, "
                       "kind TEXT NOT NULL, endpoint TEXT NOT NULL, request_hash TEXT NOT NULL, "
                       "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            db.execute("CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, stage TEXT NOT NULL, tag TEXT NOT NULL, "
                       "endpoint TEXT NOT NULL, model TEXT NOT NULL, request_hash TEXT NOT NULL, "
                       "reserved_nanos INTEGER NOT NULL, charged_nanos INTEGER, status TEXT NOT NULL, "
                       "usage_json TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
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
            raise ValueError(f"Unknown campaign stage: {stage}")

    # Preserve the public positional tag argument used by existing ledger adapters.
    # pylint: disable-next=too-many-positional-arguments
    def event(self, stage: str, kind: str, endpoint: str, digest: str, tag: str = "") -> None:
        """Persist logical requests and zero-cost local cache hits separately."""
        self._stage(stage)
        with self._transaction() as db:
            db.execute("INSERT INTO events(stage,tag,kind,endpoint,request_hash) VALUES (?,?,?,?,?)",
                       (stage, tag, kind, endpoint, digest))

    def _price(self, model: str) -> dict[str, Any]:
        models: dict[str, dict[str, Any]] = self.config["models"]
        price = models.get(model, models.get("*"))
        if price is None:
            raise CampaignLimitExceeded(f"No configured campaign pricing for model {model!r}")
        return price

    # Preserve the public positional tag argument used by existing ledger adapters.
    # pylint: disable-next=too-many-positional-arguments
    def reserve(self, stage: str, endpoint: str, payload: dict[str, Any], digest: str, tag: str = "") -> str:
        """Reserve worst-case charge atomically before a physical API attempt."""
        self._stage(stage)
        model = payload.get("model")
        if not isinstance(model, str):
            raise CampaignLimitExceeded("A priced model is required before a physical request")
        price = self._price(model)
        if endpoint != "responses" or payload.get("tools", []) or payload.get("service_tier", "default") != "default":
            raise CampaignLimitExceeded("Campaign currently permits only default-tier text Responses without tools")
        maximum = payload.get("max_output_tokens")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or not 0 < maximum <= price["max_output_tokens"]:
            raise CampaignLimitExceeded("A supported maximum output token count is required")
        messages = payload.get("input")
        if not isinstance(messages, (str, list)):
            raise CampaignLimitExceeded("Campaign requests must contain text input")
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict) or message.get("role") not in {"developer", "system", "user", "assistant"}:
                raise CampaignLimitExceeded("Only text messages are priced by the campaign guard")
            content = message.get("content")
            if isinstance(content, str):
                continue
            if (not isinstance(content, list) or any(not isinstance(part, dict)
                    or part.get("type") not in {"input_text", "output_text"}
                    or not isinstance(part.get("text"), str) for part in content)):
                raise CampaignLimitExceeded("Only text content is priced by the campaign guard")
        # Entire model input context, long-context rates, the most expensive cache
        # write tier and regional uplift give a conservative bound independent of
        # local tokenizer error, payload size, and cache warm-up.
        upper = (_rate(price["cache_write_12h_usd_per_million"]) * 2 * int(price["max_input_tokens"])
                 + _rate(price["output_usd_per_million"]) * Decimal("1.5") * maximum)
        reserved = _nanos(upper * _rate(price["regional_multiplier"]) / MILLION)
        identifier = uuid4().hex
        with self._transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            charged = "COALESCE(charged_nanos, reserved_nanos)"
            overall = int(db.execute(f"SELECT COALESCE(SUM({charged}),0) FROM attempts").fetchone()[0])
            subtotal = int(db.execute(f"SELECT COALESCE(SUM({charged}),0) FROM attempts WHERE stage=?", (stage,)).fetchone()[0])
            effective = int(db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()[0])
            if overall + reserved > effective or subtotal + reserved > self.stages[stage]:
                raise CampaignLimitExceeded(f"Campaign or stage {stage} budget would be exceeded")
            db.execute("INSERT INTO attempts(id,stage,tag,endpoint,model,request_hash,reserved_nanos,status) "
                       "VALUES (?,?,?,?,?,?,?,?)", (identifier, stage, tag, endpoint, model, digest, reserved, "reserved"))
        return identifier

    def settle(self, identifier: str, usage: dict[str, Any], payload: dict[str, Any]) -> bool:
        """Settle known token usage; leave reservations untouched when usage is unknown."""
        if not isinstance(usage.get("input_tokens"), int) or not isinstance(usage.get("output_tokens"), int):
            return False
        price = self._price(str(payload["model"]))
        total_in, output = usage["input_tokens"], usage["output_tokens"]
        details = usage.get("input_tokens_details")
        if total_in < 0 or output < 0 or output > payload["max_output_tokens"] or total_in > price["max_input_tokens"]:
            return False
        if not isinstance(details, dict) or details.get("cache_write_tokens") is None:
            cached, write30, write12 = 0, 0, total_in
        else:
            cached, writes = details.get("cached_tokens", 0) or 0, details.get("cache_write_tokens", 0) or 0
            breakdown = details.get("cache_write_tokens_details")
            if isinstance(breakdown, dict):
                write30 = breakdown.get("ttl_30m_tokens", 0) or 0
                write12 = breakdown.get("ttl_12h_tokens", 0) or 0
                if write30 + write12 != writes:
                    return False
            elif _contains_12h(payload):
                write30, write12 = 0, writes
            else:
                write30, write12 = writes, 0
        if (any(not isinstance(item, int) or item < 0 for item in (cached, write30, write12))
                or cached + write30 + write12 > total_in):
            return False
        ordinary = total_in - cached - write30 - write12
        input_multiplier = Decimal(2) if total_in > price["long_context_threshold"] else Decimal(1)
        output_multiplier = Decimal("1.5") if total_in > price["long_context_threshold"] else Decimal(1)
        cost = ((_rate(price["input_usd_per_million"]) * ordinary
                 + _rate(price["cached_input_usd_per_million"]) * cached
                 + _rate(price["cache_write_30m_usd_per_million"]) * write30
                 + _rate(price["cache_write_12h_usd_per_million"]) * write12) * input_multiplier
                + _rate(price["output_usd_per_million"]) * output * output_multiplier)
        charged = _nanos(cost * _rate(price["regional_multiplier"]) / MILLION)
        with self._transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            attempt = db.execute("SELECT reserved_nanos,status FROM attempts WHERE id=?", (identifier,)).fetchone()
            if attempt is None or attempt[1] != "reserved":
                raise ValueError("Campaign reservation is missing or already settled")
            if charged > attempt[0]:
                return False
            db.execute("UPDATE attempts SET charged_nanos=?,status='settled',usage_json=? WHERE id=?",
                       (charged, json.dumps(usage, sort_keys=True), identifier))
        return True

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
