"""Budgeted, auditable Responses transport for the Luna intent experiment.

All calls for one experiment must share the same configuration and ledger.
The ceiling cannot account for calls made outside that ledger.  A reservation
assumes the full model context, the most expensive cache write and the regional
price uplift; settled amounts are conservative estimates, not invoices.
"""

import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from datetime import timezone
from decimal import ROUND_CEILING
from decimal import Decimal
from decimal import InvalidOperation
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from carl.icl.campaign_budget import CampaignLedger
from carl.icl.luna_action_format import action_text_format

MODEL = "gpt-6-luna"
STAGE = "experiment"
CONFIG_NAME = "luna_intent_budget.json"
LEDGER_NAME = "luna_intent_ledger.sqlite3"
MODEL_PRICE = {
    "input_usd_per_million": "0.10",
    "cached_input_usd_per_million": "0.01",
    "cache_write_30m_usd_per_million": "0.125",
    "cache_write_12h_usd_per_million": "0.25",
    "output_usd_per_million": "0.50",
    "max_input_tokens": 1_050_000,
    "max_output_tokens": 128_000,
    "long_context_threshold": 272_000,
    "regional_multiplier": "1.10",
    "pricing_source": "https://developers.openai.com/api/docs/models/gpt-6-luna",
    "pricing_as_of": "2026-11-14",
}
SAFE_ERROR_CODES = frozenset({
    "authentication_error", "billing_hard_limit_reached", "content_policy_violation", "insufficient_quota",
    "invalid_api_key", "invalid_prompt", "invalid_request_error", "model_not_found", "organization_deactivated",
    "permission_denied", "permission_error", "project_deactivated", "rate_limit_exceeded", "rate_limit_error",
    "server_error", "service_unavailable", "unsupported_model", "unsupported_parameter",
})


class LunaRequestUncertain(RuntimeError):
    """A call may have incurred charges and must be reconciled before continuing."""


def _amount(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("The Luna budget must be a decimal amount") from error
    if not amount.is_finite() or not Decimal(0) < amount <= Decimal(2):
        raise ValueError("The Luna experiment budget must be greater than zero and at most $2")
    return amount


def create_luna_budget_config(run_dir: Path, limit_usd: str = "2.00") -> Path:
    """Create or reuse the immutable campaign configuration inside ``run_dir``.

    One shared run directory is essential: separate ledgers cannot enforce a
    combined limit.  This function never replaces an existing configuration.
    """
    amount = str(_amount(limit_usd))
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / CONFIG_NAME
    config = {
        "version": 1,
        "campaign_id": "luna_intent_experiment",
        "ledger_path": str(run_dir / LEDGER_NAME),
        "limit_usd": amount,
        "stage_limits_usd": {STAGE: amount},
        "models": {MODEL: MODEL_PRICE},
    }
    encoded = json.dumps(config, indent=2, sort_keys=True) + "\n"
    try:
        with config_path.open("x", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing Luna configuration differs; do not replace its ledger") from None
    return config_path


def _validate_config(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    limit = _amount(config.get("limit_usd"))
    stages = config.get("stage_limits_usd")
    if (config_path.name != CONFIG_NAME or config.get("version") != 1
            or config.get("campaign_id") != "luna_intent_experiment"
            or Path(config.get("ledger_path", "")).resolve() != config_path.parent / LEDGER_NAME
            or config.get("models") != {MODEL: MODEL_PRICE}
            or not isinstance(stages, dict) or set(stages) != {STAGE}
            or _amount(stages[STAGE]) != limit):
        raise ValueError("The Luna budget configuration is invalid or has been altered")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    with path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pace_requests(run_dir: Path) -> None:
    """Space physical requests by one second under the campaign lock."""
    path = run_dir / "last_generation.json"
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        if not boot:
            raise ValueError("Missing boot identifier")
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            elapsed = time.time() - previous["epoch"]
            if previous["boot_id"] == boot:
                elapsed = min(elapsed, time.monotonic() - previous["monotonic"])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Invalid request pacing clock")
            delay = max(0.0, 1.0 - elapsed)
        else:
            # An older process may have used the campaign before pacing existed.
            delay = 1.0
        if delay:
            time.sleep(delay)
        _atomic_json(path, {"boot_id": boot, "epoch": time.time(), "monotonic": time.monotonic()})
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise LunaRequestUncertain("Cannot verify request pacing; campaign stopped") from error


def _record_request_finished(run_dir: Path) -> None:
    """Persist the end of a physical call so the next call waits a full second."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    if not boot:
        raise LunaRequestUncertain("Cannot verify boot identity for request pacing")
    _atomic_json(run_dir / "last_generation.json", {
        "boot_id": boot, "epoch": time.time(), "monotonic": time.monotonic()
    })


def _response_dict(response: Any) -> dict[str, Any]:
    raw = response.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise ValueError("Response cannot be saved as an object")
    return raw


def _checked_usage(raw: dict[str, Any], maximum: int, *, output_text: Any = None) -> dict[str, int]:
    model = raw.get("model")
    usage = raw.get("usage")
    if (not isinstance(model, str) or not (model == MODEL or model.startswith(MODEL + "-"))
            or raw.get("service_tier") != "default" or not isinstance(usage, dict)):
        raise ValueError("Response model, service tier, or usage is unknown")
    incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
    if (not isinstance(incoming, int) or isinstance(incoming, bool) or not isinstance(outgoing, int)
            or isinstance(outgoing, bool) or not 0 <= incoming <= 1_050_000
            or not 0 <= outgoing <= maximum):
        raise ValueError("Response token counts are missing or outside the reservation")
    # Every request constructed by this transport contains a non-empty prompt.
    # Do not release a worst-case reservation when the usage counters are
    # inconsistent with that prompt or with returned output content.
    if incoming == 0 or (outgoing == 0 and (raw.get("output") or raw.get("output_text") or output_text)):
        raise ValueError("Response token counts contradict the request or output")
    # Charge all input as 12-hour cache writes; the raw breakdown stays in the audit.
    return {"input_tokens": incoming, "output_tokens": outgoing}


def _error_diagnostics(error: BaseException) -> dict[str, Any]:
    """Retain recognised diagnostics, scrubbing likely credentials from messages."""
    status = getattr(error, "status_code", None)
    result: dict[str, Any] = {"http_status": status if isinstance(status, int) and not isinstance(status, bool)
                              and 100 <= status <= 599 else None}
    for name in ("code", "type"):
        value = getattr(error, name, None)
        result[f"service_{name}"] = value if isinstance(value, str) and value in SAFE_ERROR_CODES else None
    request_id = getattr(error, "request_id", None)
    result["request_id"] = (request_id if isinstance(request_id, str) and request_id.startswith("req_")
                            and len(request_id) <= 80 and request_id[4:].isalnum() else None)
    parameter = getattr(error, "param", None)
    allowed = {"model", "input", "temperature", "reasoning", "reasoning.effort", "text", "text.format",
               "max_output_tokens", "service_tier", "tools", "store"}
    if isinstance(parameter, str) and parameter in allowed:
        result["service_param"] = parameter
    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        message = re.sub(r"(?i)https?://\S+", "[redacted-url]", message)
        message = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", message)
        message = re.sub(r"(?i)sk-[A-Za-z0-9_-]+", "[redacted-key]", message)
        message = re.sub(r"(?i)(api[_ -]?key|authorization|access[_ -]?token)\s*[:=]\s*[^\s,;}]+",
                         r"\1=[redacted]", message)
        message = re.sub(r"\b[A-Za-z0-9_/-]{24,}\b", "[redacted-token]", message)
        result["service_message"] = " ".join(message.split())[:280]
    return result


class LunaIntentTransport:
    """Sequential Luna text calls with a durable per-attempt budget and audit."""

    def __init__(self, config_path: Path, *, timeout_seconds: float = 60,
                 max_output_tokens: int = 512, client: Any = None, structured_actions: bool = False,
                 accepted_unknown_id: str | None = None) -> None:
        self.config_path = Path(config_path).resolve()
        _validate_config(self.config_path)
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (float, int))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Timeout must be a finite positive number of seconds")
        if (not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool)
                or not 0 < max_output_tokens <= 4096):
            raise ValueError("Output cap must be an integer between 1 and 4096")
        if not isinstance(structured_actions, bool):
            raise ValueError("structured_actions must be a boolean")
        if (accepted_unknown_id is not None
                and (not isinstance(accepted_unknown_id, str) or not re.fullmatch(r"[0-9a-f]{32}", accepted_unknown_id))):
            raise ValueError("accepted_unknown_id must be a reservation identifier")
        self.max_output_tokens = max_output_tokens
        self.structured_actions = structured_actions
        self.accepted_unknown_id = accepted_unknown_id
        self.ledger = CampaignLedger(self.config_path)
        self.run_dir = self.config_path.parent
        self.audit_dir = self.run_dir / "luna_requests"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        if client is None:
            from openai import OpenAI    # Imported only when the experiment needs the API.
            client = OpenAI(timeout=timeout_seconds, max_retries=0)
        else:
            client = client.with_options(timeout=timeout_seconds, max_retries=0)
        self.client = client

    def _require_clear(self) -> None:
        if self.accepted_unknown_id is not None:
            self._require_one_retained_unknown()
            return
        if any(row["status"] != "settled" for row in self.ledger.summary()["attempts"]):
            raise LunaRequestUncertain("An earlier Luna reservation is unresolved; stop and reconcile it")
        for path in self.audit_dir.glob("*.json"):
            try:
                status = json.loads(path.read_text(encoding="utf-8")).get("status")
            except (OSError, ValueError, AttributeError):
                raise LunaRequestUncertain("An audit record cannot be read; stop and reconcile it") from None
            if status != "settled":
                raise LunaRequestUncertain("An earlier Luna audit record is unresolved; stop and reconcile it")

    def _require_one_retained_unknown(self) -> None:
        """Permit exactly one specified HTTP-400 unknown without releasing it.

        The caller holds the same cross-process lock used by every request.
        SQLite is opened read-only: the unknown attempt remains unchanged and
        the ledger continues counting its entire reservation against the limit.
        """
        try:
            config_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
            with sqlite3.connect(f"{self.ledger.path.as_uri()}?mode=ro", uri=True) as database:
                database.execute("BEGIN")
                rows = database.execute(
                    "SELECT id,status,reserved_nanos,charged_nanos,stage,endpoint,model,request_hash FROM attempts"
                ).fetchall()
                metadata = dict(database.execute("SELECT key,value FROM metadata").fetchall())
            if (config_hash != self.ledger.config_hash or metadata.get("config_hash") != config_hash
                    or int(metadata["effective_limit_nanos"]) > min(self.ledger.limit, self.ledger.requested_limit)):
                raise ValueError("Budget configuration or limit changed")
            attempts = {row[0]: row for row in rows}
            outstanding = [row for row in rows if row[1] != "settled"]
            if (len(outstanding) != 1 or outstanding[0][0] != self.accepted_unknown_id
                    or outstanding[0][1] != "unknown" or outstanding[0][3] is not None):
                raise ValueError("The retained reservation is missing or another attempt is unresolved")

            audits = {}
            for path in self.audit_dir.glob("*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                identifier = record["reservation_id"]
                if identifier in audits or path.stem != identifier or identifier not in attempts:
                    raise ValueError("Audit records do not match the ledger")
                if record["status"] != attempts[identifier][1]:
                    raise ValueError("Audit and ledger statuses differ")
                audits[identifier] = record
            if set(audits) != set(attempts):
                raise ValueError("An audit record is missing")
            record = audits[self.accepted_unknown_id]
            payload = record["request"]
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            identifier, _, reserved, _, stage, endpoint, model, saved_digest = outstanding[0]
            cap = payload["max_output_tokens"]
            required_keys = {"model", "input", "max_output_tokens", "reasoning", "temperature", "service_tier", "tools", "store"}
            if (not isinstance(payload, dict) or set(payload) not in (required_keys, required_keys | {"text"})
                    or payload["model"] != MODEL or not isinstance(payload["input"], str)
                    or payload["reasoning"] != {"effort": "none"} or payload["temperature"] != 0
                    or payload["service_tier"] != "default" or payload["tools"] != [] or payload["store"] is not False
                    or ("text" in payload and payload["text"] != action_text_format())
                    or not isinstance(cap, int) or isinstance(cap, bool) or not 0 < cap <= 4096
                    or (stage, endpoint, model) != (STAGE, "responses", MODEL)
                    or digest != saved_digest or record["request_sha256"] != digest
                    or record.get("error_diagnostics", {}).get("http_status") != 400):
                raise ValueError("The retained request or its audit is inconsistent")

            price = self.ledger.config["models"][MODEL]
            bound = (Decimal(price["cache_write_12h_usd_per_million"]) * 2 * price["max_input_tokens"]
                     + Decimal(price["output_usd_per_million"]) * Decimal("1.5") * cap)
            bound *= Decimal(price["regional_multiplier"]) * Decimal(1000)
            expected = int(bound.to_integral_value(rounding=ROUND_CEILING))
            if reserved != expected or identifier != record["reservation_id"]:
                raise ValueError("The retained reservation does not match its full price bound")
        except (AttributeError, KeyError, OSError, TypeError, ValueError, sqlite3.Error):
            raise LunaRequestUncertain("The retained unknown reservation cannot be verified; campaign stopped") from None

    def create(self, prompt: str, *, stage: str = STAGE, tag: str = "") -> Any:
        """Make one physical call; no retries, caching, tools, or hidden history."""
        if not isinstance(prompt, str) or not prompt or stage != STAGE or not isinstance(tag, str):
            raise ValueError("A non-empty prompt and the experiment stage are required")
        payload = {
            "model": MODEL,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "service_tier": "default",
            "tools": [],
            "store": False,
        }
        if self.structured_actions:
            payload["text"] = action_text_format()
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        audit_tag = uuid4().hex
        with _exclusive_lock(self.run_dir / "luna_intent.lock"):
            self._require_clear()
            identifier = self.ledger.reserve(stage, "responses", payload, digest, tag=audit_tag)
            path = self.audit_dir / f"{identifier}.json"
            record: dict[str, Any] = {
                "schema_version": 1, "reservation_id": identifier, "request_sha256": digest,
                "created_at_utc": _now(), "stage": stage, "tag": tag, "status": "reserved",
                "request": payload, "response": None, "request_id": None,
                "conservative_charge_usd": None, "error_type": None,
            }
            try:
                _atomic_json(path, record)
                _pace_requests(self.run_dir)
                try:
                    response = self.client.responses.create(**payload)
                finally:
                    _record_request_finished(self.run_dir)
                raw = _response_dict(response)
                record.update(status="received", response=raw, request_id=getattr(response, "_request_id", None),
                              received_at_utc=_now())
                _atomic_json(path, record)
                usage = _checked_usage(raw, self.max_output_tokens, output_text=getattr(response, "output_text", None))
                if not self.ledger.settle(identifier, usage, payload):
                    raise ValueError("The response could not be settled")
                rows = [row for row in self.ledger.summary()["attempts"] if row["tag"] == audit_tag]
                if len(rows) != 1 or rows[0]["status"] != "settled":
                    raise ValueError("The settled charge could not be audited")
                record.update(status="settled", conservative_charge_usd=rows[0]["committed_usd"],
                              settled_at_utc=_now())
                _atomic_json(path, record)
                return response
            except BaseException as error:
                self.ledger.mark_unknown(identifier)
                record.update(status="unknown", error_type=type(error).__name__,
                              error_diagnostics=_error_diagnostics(error), failed_at_utc=_now())
                try:
                    _atomic_json(path, record)
                except OSError:
                    pass  # The durable ledger or preceding audit record still blocks further calls.
                if not isinstance(error, Exception):
                    raise
                raise LunaRequestUncertain("Luna call or accounting failed; reservation retained and campaign stopped") from None
