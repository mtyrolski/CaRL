"""Durable per-study logical and simulated physical request accounting."""
import json
import sqlite3
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from uuid import uuid4

NANOS = Decimal(1_000_000_000)


def nanos(value):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("Costs and limits must be finite and nonnegative")
    return int((number * NANOS).to_integral_value(rounding=ROUND_CEILING))


class LimitExceeded(RuntimeError):
    """A cumulative logical, physical, campaign, arm or stage limit was reached."""


class OfflineLedger:
    """SQLite reservations for mock/replay; never a replacement for the live campaign ledger."""

    def __init__(self, path: Path, limits: dict, *, create: bool = False):
        self.path, self.limits = path, limits
        if not path.exists() and not create:
            raise FileNotFoundError("Study ledger is missing; refusing to reset history")
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.transaction() as db:
                db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, job TEXT NOT NULL, attempt INTEGER NOT NULL, "
                           "stage TEXT NOT NULL, arm TEXT NOT NULL, request_hash TEXT NOT NULL, slot INTEGER NOT NULL, "
                           "status TEXT NOT NULL, reason TEXT, cache_hit INTEGER NOT NULL DEFAULT 0, "
                           "reserved_nanos INTEGER NOT NULL, charged_nanos INTEGER, usage_json TEXT, "
                           "reserved_tokens INTEGER NOT NULL DEFAULT 0)")
                contract = json.dumps(limits, sort_keys=True, separators=(",", ":"))
                db.execute("INSERT OR IGNORE INTO meta VALUES ('limits', ?)", (contract,))
                stored = db.execute("SELECT value FROM meta WHERE key='limits'").fetchone()
                if stored is None or stored[0] != contract:
                    raise ValueError("Ledger limits changed; use a new study")
        else:
            # Reopening an existing ledger must not trust the caller's limits:
            # the file is the durable accounting authority across restarts.
            with self.transaction() as db:
                stored = db.execute("SELECT value FROM meta WHERE key='limits'").fetchone()
                expected = json.dumps(limits, sort_keys=True, separators=(",", ":"))
                if stored is None or stored[0] != expected:
                    raise ValueError("Ledger limits changed; use a new study")
        with self.transaction() as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(calls)")}
        self.has_token_column = "reserved_tokens" in columns
        token_limit = limits.get("token_limit")
        if token_limit is not None and (not isinstance(token_limit, int) or isinstance(token_limit, bool) or token_limit < 0):
            raise ValueError("token_limit must be a nonnegative integer")
        if token_limit is not None and not self.has_token_column:
            raise ValueError("Token limit requires a new ledger; existing freezes are not migrated")

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, *, job: str, attempt: int, stage: str, arm: str, request_hash: str,
                slot: int, cache_hit: bool, reserve_usd: str, reserve_tokens: int = 0) -> str:
        """Atomically reserve one logical call and any physical cost before inference."""
        amount = 0 if cache_hit else nanos(reserve_usd)
        if not isinstance(reserve_tokens, int) or isinstance(reserve_tokens, bool) or reserve_tokens < 0:
            raise ValueError("Token reservation must be a nonnegative integer")
        tokens = 0 if cache_hit else reserve_tokens
        if self.limits.get("token_limit") is not None and not cache_hit and tokens == 0:
            raise ValueError("A positive offline token reservation is required before a mock request")
        identifier = uuid4().hex
        with self.transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            count, physical = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(CASE WHEN cache_hit=0 THEN 1 ELSE 0 END),0) FROM calls"
            ).fetchone()
            if count >= self.limits["logical_calls"]:
                raise LimitExceeded("logical_calls")
            if not cache_hit and physical >= self.limits["physical_calls"]:
                raise LimitExceeded("physical_calls")
            if self.limits.get("token_limit") is not None:
                committed_tokens = db.execute("SELECT COALESCE(SUM(reserved_tokens),0) FROM calls").fetchone()[0]
                if committed_tokens + tokens > self.limits["token_limit"]:
                    raise LimitExceeded("token_limit")
            total = db.execute("SELECT COALESCE(SUM(COALESCE(charged_nanos,reserved_nanos)),0) FROM calls").fetchone()[0]
            arm_total = db.execute("SELECT COALESCE(SUM(COALESCE(charged_nanos,reserved_nanos)),0) FROM calls WHERE arm=?", (arm,)).fetchone()[0]
            stage_total = db.execute("SELECT COALESCE(SUM(COALESCE(charged_nanos,reserved_nanos)),0) FROM calls WHERE stage=?", (stage,)).fetchone()[0]
            if (total + amount > nanos(self.limits["total_usd"])
                    or arm_total + amount > nanos(self.limits["arm_usd"][arm])
                    or stage_total + amount > nanos(self.limits["stage_usd"][stage])):
                raise LimitExceeded("usd_limit")
            if self.has_token_column:
                db.execute("INSERT INTO calls(id,job,attempt,stage,arm,request_hash,slot,status,cache_hit,reserved_nanos,reserved_tokens) "
                           "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (identifier, job, attempt, stage, arm, request_hash, slot, "reserved", int(cache_hit), amount, tokens))
            else:
                db.execute("INSERT INTO calls(id,job,attempt,stage,arm,request_hash,slot,status,cache_hit,reserved_nanos) "
                           "VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (identifier, job, attempt, stage, arm, request_hash, slot, "reserved", int(cache_hit), amount))
        return identifier

    def finish(self, identifier: str, *, status: str, charged_usd: str | None, usage: dict, reason: str | None = None):
        if status not in {"settled", "unknown", "cache_hit", "replay_miss", "cache_error"}:
            raise ValueError("Invalid reservation status")
        if (status == "unknown") != (charged_usd is None):
            raise ValueError("Unknown billing must retain its reservation; known billing needs a charge")
        amount = None if charged_usd is None else nanos(charged_usd)
        with self.transaction() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved_nanos,status FROM calls WHERE id=?", (identifier,)).fetchone()
            if row is None or row[1] != "reserved" or (amount is not None and amount > row[0]):
                raise ValueError("Reservation missing, settled or exceeded")
            db.execute("UPDATE calls SET status=?,charged_nanos=?,usage_json=?,reason=? WHERE id=?",
                       (status, amount, json.dumps(usage, sort_keys=True), reason, identifier))

    def summary(self) -> dict:
        with self.transaction() as db:
            rows = db.execute("SELECT stage,arm,status,COUNT(*),SUM(COALESCE(charged_nanos,reserved_nanos)) "
                              "FROM calls GROUP BY stage,arm,status ORDER BY stage,arm,status").fetchall()
            unresolved = db.execute("SELECT COUNT(*) FROM calls WHERE status IN ('reserved','unknown')").fetchone()[0]
            total = db.execute("SELECT COUNT(*),COALESCE(SUM(CASE WHEN status='cache_hit' THEN 1 ELSE 0 END),0),"
                               "COALESCE(SUM(CASE WHEN cache_hit=0 THEN 1 ELSE 0 END),0) FROM calls").fetchone()
            tokens = db.execute("SELECT COALESCE(SUM(reserved_tokens),0) FROM calls").fetchone()[0] if self.has_token_column else 0
        return {"logical_calls": total[0], "local_cache_hits": total[1],
                "mock_new_calls": total[2], "unresolved_reservations": unresolved,
                "provider_physical_requests": 0, "provider_prompt_cached_input_tokens": None,
                "actual_provider_cost_usd": 0.0,
                "offline_token_limit": self.limits.get("token_limit"), "offline_token_reservations": tokens,
                "by_stage_arm_status": [{"stage": s, "arm": a, "status": st, "calls": n,
                                          "committed_usd": amount / 1e9} for s, a, st, n, amount in rows],
                "note": "Offline simulated ledger only; physical provider requests and real cost are zero."}
