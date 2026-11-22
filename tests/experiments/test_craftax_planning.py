"""Run the four shared planning modes against a native Craftax transition stream."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from carl.icl.config import ROOT
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.types import JSONObject
from scripts import run_predicate_icl


@pytest.mark.skipif(importlib.util.find_spec("craftax") is None, reason="optional pinned Craftax runtime")
def test_native_craftax_four_modes_replay_one_matched_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "matched.json"
    task_path = ROOT / "configs/icl/craftax/tasks.json"
    roles = set()
    call = ExperimentAPI.call

    def record(api: ExperimentAPI, endpoint: str, payload: JSONObject, *args: Any, **kwargs: Any) -> JSONObject:
        query = json.loads(payload["input"][0]["content"][0]["text"])
        private = {"_state", "_key", "seed", "fingerprint", "chain", "snapshot"}
        assert private.isdisjoint(query) and private.isdisjoint(query["observation"])
        roles.add(payload["metadata"]["operation"])
        return call(api, endpoint, payload, *args, **kwargs)

    monkeypatch.setattr(ExperimentAPI, "call", record)
    monkeypatch.setattr(sys, "argv", [
        "run_predicate_icl", "craftax", "--tasks", str(task_path), "--task", "wood-to-stone-pickaxe",
        "--mode", "mock", "--planning-mode", "all", "--output", str(path),
    ])
    assert run_predicate_icl.main() == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data["results"]
    assert roles == {"action", "intent", "value", "plan"}
    assert set(results) == {"direct", "one-step", "full-search", "plan-and-act"}
    assert data["execution_mode"] == "mock" and data["preparation"]["attempted_simulator_actions"] == 1
    assert data["policy"]["fingerprint"] and data["task_manifest_sha256"]
    assert not {"_state", "_key", "seed", "fingerprint"} & data["initial_observation"].keys()
    expected_actions = [5, 11, 4, 5, 12]
    expected_rewards = [1.0, 1.0, 0.0, 1.0, 1.0]
    for result in results.values():
        assert result["solved"] and result["action_path"] == expected_actions
        assert result["nodes"][0]["state"]["snapshot"] == data["initial_snapshot"]
        assert result["endpoint"]["truncated"] and not result["endpoint"]["terminated"]
        assert result["accounting"].keys() == results["direct"]["accounting"].keys()
        assert result["accounting"]["physical_requests"] == 0
        assert result["accounting"]["executed_plan_actions"] == 5
        assert {"_state", "_key"}.isdisjoint(result["endpoint"])
        mapped = {attempt["node"]: attempt for attempt in result["attempts"] if attempt["node"] is not None}
        steps = []
        node = result["selected_node"]
        while node:
            steps.insert(0, mapped[node]["rewards"])
            node = mapped[node]["parent"]
        assert [reward for step in steps for reward in step] == expected_rewards
    assert results["one-step"]["accounting"]["attempted_simulator_actions"] > 5
    assert results["plan-and-act"]["accounting"]["logical_generator_calls"] == 1
    assert len({result["endpoint"]["fingerprint"] for result in results.values()}) == 1
