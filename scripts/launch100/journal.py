"""Atomic response checkpoints and deterministic trace replay for interrupted runs."""

import json
import os
from collections.abc import Callable
from pathlib import Path

from carl.icl.api import RequestFailed, request_cache_digest, write_json
from carl.icl.config import ExperimentConfig
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.types import JSONObject
from scripts.launch100.plan import digest
from scripts.launch100.retry import call_with_transport_retry, has_completed_response


class EventJournal:
    """Verify an existing event prefix, then durably append new actual transitions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.saved = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        self.index = 0

    def __call__(self, event: JSONObject) -> None:
        event = json.loads(json.dumps(event))
        if self.index < len(self.saved):
            if self.saved[self.index] != event:
                raise ValueError(f"Resume trace diverged at event {self.index}")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        self.index += 1

    def verify_consumed(self) -> None:
        """A shorter replay is a protocol failure, not a completed new run."""
        if self.index < len(self.saved):
            raise ValueError("Resume did not consume the saved trace")


class JournalAPI(ExperimentAPI):
    """Replay exactly saved responses, then resume through the normal cache/ledger.

    Saved cumulative metrics preserve logical limits across restarts. An uncertain
    in-flight request must already have a completed cache entry before resuming;
    it is never silently repeated against the provider.
    """

    def __init__(self, config: ExperimentConfig, directory: Path, tag: str) -> None:
        super().__init__(config.path("cache_dir"), "openai", config.limits)
        self.budget_tag = tag
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = sorted(directory.glob("call_*.json"))
        if [path.name for path in self.paths] != [f"call_{index:06d}.json" for index in range(len(self.paths))]:
            raise ValueError("Response checkpoint sequence has gaps")
        self.index = 0
        self.replayed_calls = 0
        self.reconciled_pending_calls = 0
        self.pending = directory / "pending.json"
        if self.pending.exists():
            pending = json.loads(self.pending.read_text())
            if pending["index"] < len(self.paths):
                saved = json.loads(self.paths[pending["index"]].read_text())
                if saved["request_hash"] != pending["request_hash"]:
                    raise ValueError("Pending marker conflicts with completed response")
                self.pending.unlink()

    def call(
        self, endpoint: str, payload: JSONObject, mock: Callable[[], JSONObject] | None = None,
    ) -> JSONObject:
        request = {"endpoint": endpoint, "payload": payload}
        key = request_cache_digest(request)
        if self.index < len(self.paths):
            saved = json.loads(self.paths[self.index].read_text())
            if saved["request_hash"] != key or saved["response_sha256"] != digest(saved["response"]):
                raise ValueError(f"Response checkpoint differs at call {self.index}")
            self.metrics.update(saved["metrics"])
            self.index += 1
            self.replayed_calls += 1
            return saved["response"]
        recovering = self.pending.exists()
        if recovering:
            pending = json.loads(self.pending.read_text())
            if pending != {"index": self.index, "request_hash": key}:
                raise ValueError("Interrupted request differs from deterministic replay")
            # No network is permitted while reconciling an uncertain request.
            if not has_completed_response(self, endpoint, payload):
                raise RequestFailed("Interrupted request lacks a completed cache record; inspect shared ledger")
        else:
            write_json(self.pending, {"index": self.index, "request_hash": key})
        previous = self.provider
        if recovering:
            self.provider = "replay"
        try:
            result = call_with_transport_retry(self, endpoint, payload, mock, call=super().call)
        finally:
            self.provider = previous
        # Complete provider payloads/outputs remain in the shared content-addressed cache.
        response = {name: result[name] for name in ("output_text", "usage", "model", "status", "response_id")
                    if name in result}
        write_json(self.directory / f"call_{self.index:06d}.json",
                   {"request_hash": key, "response": response, "response_sha256": digest(response),
                    "metrics": dict(self.metrics)})
        self.pending.unlink()
        self.index += 1
        self.reconciled_pending_calls += int(recovering)
        return result

    def verify_consumed(self) -> None:
        """Prevent accepting results that truncate an earlier response history."""
        if self.index < len(self.paths):
            raise ValueError("Resume did not consume saved responses")
