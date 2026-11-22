"""Persistent budget for the offline Sokoban and Craftax pilot."""

import json
import math
import os
import signal
import sqlite3
import subprocess
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


LIMIT_MS = 7_200_000
PROMPT_09_MS = 1_800_000
PHASES = ("preparation", "inference")
PREPARATION = ("model_download", "human_annotation", "codex_annotation", "codex_usage")
DOMAINS = ("sokoban", "craftax", "both")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PilotLedger:
    """Reserve cumulative worker time with SQLite's cross-process write lock."""

    def __init__(self, root: Path, *, create: bool = False) -> None:
        root = root.resolve()
        self.path = root / "campaign.sqlite3"
        if create:
            root.mkdir(parents=True, exist_ok=True)
            self._initialize()
        if not self.path.is_file():
            raise ValueError(f"Initialize the pilot ledger at {self.path} first")

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=15)) as conn, conn:
            conn.row_factory = sqlite3.Row
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn

    def _initialize(self) -> None:
        with closing(sqlite3.connect(self.path, timeout=15)) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS contract (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, prompt INTEGER NOT NULL,
                    domain TEXT NOT NULL, phase TEXT NOT NULL, activity TEXT NOT NULL,
                    command TEXT NOT NULL, started_utc TEXT NOT NULL,
                    workers INTEGER NOT NULL, reserved_ms INTEGER NOT NULL,
                    charged_ms INTEGER, wall_ms INTEGER,
                    status TEXT NOT NULL, exit_code INTEGER
                );
                CREATE TABLE IF NOT EXISTS preparation (
                    id TEXT PRIMARY KEY, prompt INTEGER NOT NULL, activity TEXT NOT NULL,
                    seconds REAL, input_tokens INTEGER, output_tokens INTEGER,
                    status TEXT NOT NULL, note TEXT NOT NULL, recorded_utc TEXT NOT NULL
                );
            """)
            budget = {
                "schema": "intent-pilot-budget-v1", "metered_usd": "0",
                "local_limit_ms": str(LIMIT_MS), "prompt_09_ms": str(PROMPT_09_MS),
            }
            conn.executemany("INSERT OR IGNORE INTO contract VALUES (?, ?)", budget.items())
            if any(conn.execute("SELECT value FROM contract WHERE key = ?", (key,)).fetchone()[0] != value
                   for key, value in budget.items()):
                raise ValueError("The existing pilot ledger uses a different budget contract")
            rows = (
                ("initial-model-download", "model_download", 0.0, "not_started", "Inspected local files"),
                ("initial-human-annotation", "human_annotation", 0.0, "not_started", "Split freeze only"),
                ("initial-codex-annotation", "codex_annotation", 0.0, "not_started", "Split freeze only"),
                ("initial-codex-usage", "codex_usage", None, "unknown", "Per-run Codex usage is unavailable"),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO preparation VALUES (?, 1, ?, ?, NULL, NULL, ?, ?, ?)",
                [(key, activity, seconds, status, note, _now()) for key, activity, seconds, status, note in rows],
            )

    def bind_split(self, digest: str) -> None:
        """Bind once to the immutable split package."""
        with self._connection(write=True) as conn:
            conn.execute("INSERT OR IGNORE INTO contract VALUES ('split_sha256', ?)", (digest,))
            if conn.execute("SELECT value FROM contract WHERE key = 'split_sha256'").fetchone()[0] != digest:
                raise ValueError("This campaign is already bound to another split package")

    def split_digest(self) -> str | None:
        with self._connection() as conn:
            row = conn.execute("SELECT value FROM contract WHERE key = 'split_sha256'").fetchone()
            return str(row[0]) if row else None

    def bind_candidate_pool(self, version: int, digest: str) -> None:
        """Record a pool version without allowing an existing edition to change."""
        key = f"candidate_pool_v{version}_sha256"
        with self._connection(write=True) as conn:
            conn.execute("INSERT OR IGNORE INTO contract VALUES (?, ?)", (key, digest))
            if conn.execute("SELECT value FROM contract WHERE key = ?", (key,)).fetchone()[0] != digest:
                raise ValueError("This campaign is already bound to a different candidate pool for this version")

    def candidate_pool_digest(self, version: int) -> str | None:
        with self._connection() as conn:
            row = conn.execute("SELECT value FROM contract WHERE key = ?", (f"candidate_pool_v{version}_sha256",)).fetchone()
            return str(row[0]) if row else None

    def _reserve(
        self, prompt: int, domain: str, phase: str, activity: str,
        command: list[str], seconds: int, workers: int,
    ) -> str:
        if not 1 <= prompt <= 9 or domain not in DOMAINS or phase not in PHASES:
            raise ValueError("Invalid prompt, domain or phase")
        if min(seconds, workers) < 1 or not command:
            raise ValueError("Specify a positive timeout and worker count and a local command")
        # Include one worker-second of cleanup headroom per worker.
        reservation = (seconds + 1) * 1_000 * workers
        with self._connection(write=True) as conn:
            total, early = conn.execute("""
                SELECT COALESCE(SUM(COALESCE(charged_ms, reserved_ms)), 0),
                       COALESCE(SUM(CASE WHEN prompt < 9 THEN COALESCE(charged_ms, reserved_ms) END), 0)
                FROM jobs
            """).fetchone()
            if total + reservation > LIMIT_MS or (prompt < 9 and early + reservation > LIMIT_MS - PROMPT_09_MS):
                raise ValueError("Local experiment time exceeds the available budget")
            job_id = uuid4().hex
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 'reserved', NULL)",
                (job_id, prompt, domain, phase, activity, json.dumps(command), _now(), workers, reservation),
            )
        return job_id

    def _settle(self, job_id: str, status: str, code: int, wall_ms: int, workers: int) -> None:
        with self._connection(write=True) as conn:
            conn.execute(
                "UPDATE jobs SET status=?, exit_code=?, wall_ms=?, charged_ms=? WHERE id=? AND status='reserved'",
                (status, code, wall_ms, wall_ms * workers, job_id),
            )

    def run(
        self, prompt: int, domain: str, phase: str, activity: str,
        command: list[str], *, seconds: int, workers: int,
    ) -> int:
        """Charge failures, retries and timeouts as well as successful local commands."""
        job_id = self._reserve(prompt, domain, phase, activity, command, seconds, workers)
        env = {
            **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "OPENAI_API_KEY": "", "MISTRAL_API_KEY": "",
        }
        start = time.monotonic_ns()
        try:
            # GNU timeout owns the command group and enforces its deadline even if this process dies.
            code = subprocess.run(
                ["timeout", "--signal=TERM", "--kill-after=0.5s", f"{seconds}s", *command],
                env=env, check=False,
            ).returncode
        except OSError:
            self._settle(job_id, "launch_failed", 127, _elapsed_ms(start), workers)
            raise
        except KeyboardInterrupt:
            # Timeout can still be stopping its group. Preserve the reservation.
            raise
        status = "complete" if code == 0 else "timed_out" if code in (124, -signal.SIGKILL) else "failed"
        self._settle(job_id, status, code, _elapsed_ms(start), workers)
        return code if code >= 0 else 128 - code

    def record_preparation(
        self, prompt: int, activity: str, status: str, note: str,
        *, seconds: float | None, input_tokens: int | None, output_tokens: int | None,
    ) -> None:
        if not 1 <= prompt <= 9 or activity not in PREPARATION:
            raise ValueError("Unknown preparation category or prompt")
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Preparation seconds must be finite and non-negative")
        if any(tokens is not None and tokens < 0 for tokens in (input_tokens, output_tokens)):
            raise ValueError("Token counts must be non-negative")
        with self._connection(write=True) as conn:
            conn.execute(
                "INSERT INTO preparation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uuid4().hex, prompt, activity, seconds, input_tokens, output_tokens, status, note, _now()),
            )

    def status(self) -> dict:
        with self._connection() as conn:
            jobs = [dict(row) for row in conn.execute("SELECT * FROM jobs ORDER BY rowid")]
            preparation = [dict(row) for row in conn.execute("SELECT * FROM preparation ORDER BY rowid")]
        settled = {phase: 0 for phase in PHASES}
        held = early = final = 0
        for job in jobs:
            charged = job["charged_ms"]
            if charged is None:
                committed = job["reserved_ms"]
                held += committed
            else:
                committed = charged
                settled[job["phase"]] += committed
            if job["prompt"] < 9:
                early += committed
            else:
                final += committed
        remaining = max(0, LIMIT_MS - early - final)
        return {
            "ledger": str(self.path), "metered_usd_limit": 0, "reported_metered_usd": 0,
            "local_limit_seconds": LIMIT_MS / 1_000,
            "settled_worker_seconds": {phase: ms / 1_000 for phase, ms in settled.items()},
            "outstanding_worker_seconds": held / 1_000,
            "remaining_worker_seconds": remaining / 1_000,
            "remaining_before_prompt_09_seconds": min(remaining, max(0, LIMIT_MS - PROMPT_09_MS - early)) / 1_000,
            "prompt_09_protected_seconds": max(0, PROMPT_09_MS - final) / 1_000,
            "jobs": jobs, "separate_preparation": preparation,
        }


def _elapsed_ms(start_ns: int) -> int:
    return (time.monotonic_ns() - start_ns + 999_999) // 1_000_000
