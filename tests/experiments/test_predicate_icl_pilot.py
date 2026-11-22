"""Exercise held-out predicate prompts, frozen support and offline parsing."""

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from carl.icl.api import BudgetExceeded
from carl.icl.config import ExperimentConfig
from carl.icl.experiment import create_inference_api
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.intent_annotations import AnnotationRequest, mock_annotator, prepare_annotations
from carl.icl.intent_domain import TASK_GOAL, Intent, SokobanObservation, decode_intent, render_instruction
from carl.icl.intents.generation.data import read_records
from carl.icl.llm_components.predicate import create_predicate_components
from carl.icl.policy_relative.measurements import MeasurementSettings, measure
from carl.icl.types import JSONObject
from scripts import measure_intent_policy, run_predicate_icl


RIGHT = '{"action":"RIGHT"}'
PROPOSAL = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": [], "instruction": None}


@dataclass(frozen=True)
class PilotData:
    path: Path
    training: tuple[JSONObject, ...]
    validation: JSONObject
    held_out: JSONObject

    @property
    def query(self) -> JSONObject:
        return {"observation": {"rows": self.held_out["start_observation"]},
                "task_goal": self.held_out["task_goal"], "scale": None, "split": "test",
                "task_group": self.held_out["source_id"]}


def _json(path: Path, content: object) -> Path:
    path.write_text(json.dumps(content) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path, rows: list[JSONObject]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture(name="pilot")
def fixture_pilot(tmp_path: Path) -> PilotData:
    board_specs = (
        ("train-01", "train", "#-----#"),
        ("train-02", "train", "#--#--#"),
        ("train-03", "train", "#-##--#"),
        ("train-04", "train", "#-###-#"),
        ("validation-01", "validation", "#-#####"),
        ("test-01", "test", "#######"),
    )
    trajectories = [
        {"source_id": source, "split": split, "actions": [3],
         "start_observation": ["#######", "#@$-.-#", "#-$-.-#", "#-----#", walls, "#-----#", "#######"]}
        for source, split, walls in board_specs
    ]

    def annotate(pair: AnnotationRequest) -> Intent:
        if pair.start_observation.rows[4] == "#######":
            # A distinct secret label for the test push, also checked by the simulator.
            return decode_intent({"effects": [{"predicate": "player_at", "row": 1, "col": 2}],
                                  "preserve": [], "instruction": None})
        return mock_annotator(pair)

    source = _json(tmp_path / "demonstrations.json", {"trajectories": trajectories})
    output = tmp_path / "annotations"
    stats = prepare_annotations(source, output, max_offset=1, seed=17, annotator=annotate)
    assert stats["accepted"] == 6
    path = output / "D_A.jsonl"
    rows = read_records(path)
    training = tuple(row for row in rows if row["split"] == "train")
    validation, = [row for row in rows if row["split"] == "validation"]
    held_out, = [row for row in rows if row["split"] == "test"]
    assert len({row["layout_id"] for row in rows}) == 6
    assert all(row["intent"] != held_out["intent"] for row in (*training, validation))
    return PilotData(path, training, validation, held_out)


def _config(tmp_path: Path, pilot: PilotData, shots: int, *, arm: str = "unscaled", **changes: Any) -> Path:
    return _json(tmp_path / "config.json", {
        "model": "frozen-fixture-model",
        "seed": 73, "generator_tokens": 256, "action_budget": 1,
        "cache_dir": str(tmp_path / "cache"),
        "predicate_icl": {
            "policy_shots": shots, "generator_shots": shots, "context_limit": 8192,
            "validation_id": pilot.validation["record_id"], "scale_arm": arm, **changes,
        },
    })


@pytest.fixture(name="cli")
def fixture_cli(monkeypatch: pytest.MonkeyPatch) -> Callable[..., int]:
    def invoke(command: str, config: Path, data: Path, manifest: Path, *args: str) -> int:
        monkeypatch.setattr(sys, "argv", ["run_predicate_icl", command, "--config", str(config),
                                         "--data", str(data), "--manifest", str(manifest), *args])
        return run_predicate_icl.main()

    return invoke


def _message(message: JSONObject) -> str:
    return "".join(part["text"] for part in message["content"])


def _validation_args(
    cli: Callable[..., int], config: Path, data: Path, manifest: Path, pilot: PilotData,
    *, tmp_path: Path, scale: float | None = None, data_args: tuple[str, ...] = (),
) -> tuple[str, ...]:
    query = _json(tmp_path / "validation-query.json", {
        "observation": {"rows": pilot.validation["start_observation"]},
        "task_goal": TASK_GOAL, "scale": scale, "split": "validation",
    })
    receipt = tmp_path / "validation-run.json"
    mock = _json(tmp_path / "mock.json", {"policy": RIGHT, "generator": json.dumps(PROPOSAL)})
    mock_args = ("--mode", "mock", "--mock-response", str(mock))
    assert cli("run", config, data, manifest, *data_args, *mock_args,
               "--query", str(query), "--output", str(receipt)) == 0
    assert _read_json(receipt)["status"] == "achieved"
    return *data_args, *mock_args, "--validation-result", str(receipt)


@pytest.mark.parametrize("shots", [0, 4])
def test_keyless_preview_and_mock_use_the_same_explicit_role_prompts(
    tmp_path: Path, pilot: PilotData, cli: Callable[..., int], monkeypatch: pytest.MonkeyPatch, shots: int,
) -> None:
    for env in ("OPENAI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    requests: dict[str, JSONObject] = {}
    original_call = ExperimentAPI.call

    def record(api: ExperimentAPI, endpoint: str, payload: JSONObject, *args: Any, **kwargs: Any) -> JSONObject:
        requests[payload["metadata"]["component"]] = payload
        return original_call(api, endpoint, payload, *args, **kwargs)

    monkeypatch.setattr(ExperimentAPI, "call", record)
    config = _config(tmp_path, pilot, shots)
    manifest = tmp_path / "support.json"
    assert cli("freeze", config, pilot.path, manifest) == 0
    query = _json(tmp_path / "query.json", pilot.query)
    proposal = _json(tmp_path / "separate-preview-intent.json", PROPOSAL)
    preview = tmp_path / "preview.json"
    assert cli("preview", config, pilot.path, manifest, "--query", str(query), "--intent", str(proposal),
               "--output", str(preview)) == 0
    assert not requests
    assert not (tmp_path / "cache").exists()
    rendered = _read_json(preview)
    assert rendered["status"] == "preview"
    assert {role: contract["shots"] for role, contract in rendered["components"].items()} == {
        "policy": shots, "generator": shots,
    }

    args = _validation_args(cli, config, pilot.path, manifest, pilot, tmp_path=tmp_path)
    requests.clear()
    result = tmp_path / "run.json"
    assert cli("run", config, pilot.path, manifest, "--query", str(query), *args, "--output", str(result)) == 0
    assert set(requests) == {"policy", "generator"}
    run = _read_json(result)
    assert run["status"] == "achieved" and run["actions"] == [3]
    assert run["intent"]["effects"] == PROPOSAL["effects"]
    assert run["support_fingerprint"] == rendered["support_fingerprint"]
    assert run["components"] == rendered["components"]
    assert run["metrics"]["physical_requests"] == 0
    assert {payload["model"] for payload in requests.values()} == {"frozen-fixture-model"}
    assert _message(requests["policy"]["input"][0]) != _message(requests["generator"]["input"][0])
    for role, payload in requests.items():
        messages = payload["input"]
        assert messages == rendered["prompts"][role]["input"]
        assert [message["role"] for message in messages] == ["developer", *["user", "assistant"] * shots, "user"]
        assert pilot.held_out["instruction"] not in json.dumps(messages)
        assistants = [json.loads(_message(message)) for message in messages if message["role"] == "assistant"]
        if role == "policy":
            assert assistants == [{"action": "RIGHT"}] * shots
        else:
            assert assistants == [pilot.training[0]["intent"]] * shots
            assert all(json.loads(_message(message))["scale"] is None for message in messages[1::2])
    generator_query = json.loads(_message(requests["generator"]["input"][-1]))
    assert set(generator_query) == {"observation", "task_goal", "scale"}
    assert generator_query["task_goal"] == TASK_GOAL and generator_query["scale"] is None
    policy_query = json.loads(_message(requests["policy"]["input"][-1]))
    assert set(policy_query) == {"observation", "instruction", "attempt_history"}
    assert policy_query["instruction"] == render_instruction(decode_intent(PROPOSAL))


def test_support_repeats_and_rejects_changed_manifest_or_source(
    tmp_path: Path, pilot: PilotData, cli: Callable[..., int], capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, pilot, 4)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert cli("freeze", config, pilot.path, first) == 0
    assert cli("freeze", config, pilot.path, second) == 0
    assert first.read_bytes() == second.read_bytes()
    frozen = first.read_text(encoding="utf-8")
    assert all(row["record_id"] in frozen for row in pilot.training)
    assert pilot.held_out["record_id"] not in frozen

    query = _json(tmp_path / "query.json", pilot.query)
    preview = ("--query", str(query), "--intent", str(_json(tmp_path / "intent.json", PROPOSAL)))
    selected_id = pilot.training[0]["record_id"]
    first.write_text(frozen.replace(selected_id, "0" * len(selected_id)), encoding="utf-8")
    with pytest.raises(SystemExit) as failed:
        cli("preview", config, pilot.path, first, *preview)
    assert failed.value.code == 1
    assert "support" in capsys.readouterr().err.lower()

    rows = read_records(pilot.path)
    for row in rows:
        if row["record_id"] == selected_id:
            row["source_id"] += "-changed"
    _jsonl(pilot.path, rows)
    with pytest.raises(SystemExit) as failed:
        cli("preview", config, pilot.path, second, *preview)
    assert failed.value.code == 1
    assert "support" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("leak", ["reflection", "task_group", "target_field"])
def test_preview_rejects_held_out_leaks(
    tmp_path: Path, pilot: PilotData, cli: Callable[..., int], capsys: pytest.CaptureFixture[str], leak: str,
) -> None:
    config = _config(tmp_path, pilot, 4)
    manifest = tmp_path / "support.json"
    assert cli("freeze", config, pilot.path, manifest) == 0
    query = pilot.query
    if leak == "reflection":
        rows = pilot.training[0]["start_observation"]
        query["observation"] = {"rows": [row[::-1] for row in rows]}
        assert query["observation"]["rows"] != rows
    elif leak == "task_group":
        query["task_group"] = pilot.training[0]["source_id"]
    else:
        query["intent"] = pilot.held_out["intent"]

    with pytest.raises(SystemExit) as failed:
        cli("preview", config, pilot.path, manifest, "--query", str(_json(tmp_path / "query.json", query)),
            "--intent", str(_json(tmp_path / "intent.json", PROPOSAL)))
    assert failed.value.code == 1
    expected = {"reflection": "canonical layout", "task_group": "task group", "target_field": "intent"}
    assert expected[leak] in capsys.readouterr().err.lower()


def test_freeze_rejects_a_task_group_shared_by_different_held_out_layouts(
    tmp_path: Path, pilot: PilotData, cli: Callable[..., int], capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, pilot, 0)
    rows = read_records(pilot.path)
    for row in rows:
        if row["record_id"] in {pilot.training[0]["record_id"], pilot.held_out["record_id"]}:
            row["task_group_id"] = "shared-across-splits"
    _jsonl(pilot.path, rows)
    manifest = tmp_path / "support.json"
    with pytest.raises(SystemExit) as failed:
        cli("freeze", config, pilot.path, manifest)
    assert failed.value.code == 1
    assert "task group" in capsys.readouterr().err.lower()
    assert not manifest.exists()


def test_invalid_policy_output_cannot_authorize_the_held_out_run(
    tmp_path: Path, pilot: PilotData, cli: Callable[..., int], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, pilot, 0)
    manifest = tmp_path / "support.json"
    assert cli("freeze", config, pilot.path, manifest) == 0
    query = _json(tmp_path / "validation-query.json", {
        "observation": {"rows": pilot.validation["start_observation"]},
        "task_goal": TASK_GOAL, "split": "validation",
    })
    mock = _json(tmp_path / "bad-policy.json", {
        "policy": '{"action":"TELEPORT"}', "generator": json.dumps(PROPOSAL),
    })
    receipt = tmp_path / "validation.json"
    assert cli("run", config, pilot.path, manifest, "--query", str(query), "--mode", "mock",
               "--mock-response", str(mock), "--output", str(receipt)) == 2
    result = _read_json(receipt)
    assert result["status"] == "invalid_response" and not result["actions"]
    assert "validation_result" not in result

    def unexpected_call(*_args: Any, **_kwargs: Any) -> JSONObject:
        pytest.fail("Held-out query reached the API without valid validation")

    monkeypatch.setattr(ExperimentAPI, "call", unexpected_call)
    query = _json(tmp_path / "test-query.json", pilot.query)
    with pytest.raises(SystemExit) as failed:
        cli("run", config, pilot.path, manifest, "--query", str(query), "--mode", "mock",
            "--mock-response", str(mock), "--validation-result", str(receipt))
    assert failed.value.code == 1
    assert "successful validation" in capsys.readouterr().err


def test_annotation_source_preserves_task_groups_and_rejects_overlap(tmp_path: Path, pilot: PilotData) -> None:
    rows = [pilot.training[0], pilot.held_out]
    trajectories = [{**{key: row[key] for key in ("source_id", "start_observation", "split")},
                     "actions": [3], "task_group": "shared"} for row in rows]
    source = _json(tmp_path / "grouped.json", {"trajectories": trajectories})
    output = tmp_path / "grouped"
    with pytest.raises(ValueError, match="Task group overlaps"):
        prepare_annotations(source, output, max_offset=1, seed=17)
    trajectories[1]["task_group"] = "held-out"
    _json(source, {"trajectories": trajectories})
    prepare_annotations(source, output, max_offset=1, seed=17)
    assert {row["task_group"] for row in read_records(output / "D_A.jsonl")} == {"shared", "held-out"}


def test_calibration_cli_rejects_layouts_used_as_policy_support(
    tmp_path: Path, pilot: PilotData, cli: Callable[..., int], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, pilot, 4)
    manifest = tmp_path / "support.json"
    output = tmp_path / "measurements"
    assert cli("freeze", config, pilot.path, manifest) == 0
    monkeypatch.setattr(sys, "argv", [
        "measure_intent_policy", "--mock", "--predicate-config", str(config),
        "--support-manifest", str(manifest), "--data", str(pilot.path), "--output", str(output),
    ])
    with pytest.raises(SystemExit) as failed:
        measure_intent_policy.main()
    assert failed.value.code == 1
    assert "overlaps policy support" in capsys.readouterr().err
    assert not output.exists()


def test_controller_scales_from_the_stage04_loader_reach_generator_support(
    tmp_path: Path, pilot: PilotData, cli: Callable[..., int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "measurement"
    calibration_config = ExperimentConfig(cache_dir=str(tmp_path / "calibration-cache"))
    api = create_inference_api(calibration_config, "mock", encoding=ByteEncoding())
    components = create_predicate_components(calibration_config, api, context_limit=8192, scale_arm="unscaled")
    policy = components.contract("policy")
    summary = measure(pilot.path, output, settings=MeasurementSettings(seed=31, action_cap=4),
                      controller={"id": "predicate-policy-fixture", "deterministic": True, "predicate_policy": policy},
                      policy_factory=lambda seed: components.policy_factory(seed, mock=lambda _payload: RIGHT))
    assert summary["status"] == "complete"
    assert api.metrics["policy_calls"] == 5 and api.metrics["physical_requests"] == 0
    data = output / "D_C.jsonl"
    for row in read_records(data):
        contract = row["policy_measurements"]["controller_contract"]
        assert contract["prompt_sha256"] == policy["prompt_sha256"]
        assert contract["controller"]["predicate_policy"] == policy
    config = _config(tmp_path, pilot, 4, arm="relative_u", validation_scale=0.5)
    manifest = tmp_path / "support.json"
    assert cli("freeze", config, data, manifest, "--policy-data", str(pilot.path)) == 0
    query = _json(tmp_path / "query.json", {**pilot.query, "scale": 0.7})
    args = _validation_args(cli, config, data, manifest, pilot, tmp_path=tmp_path, scale=0.5,
                            data_args=("--policy-data", str(pilot.path)))
    scales: list[float] = []
    original_call = ExperimentAPI.call

    def record(api: ExperimentAPI, endpoint: str, payload: JSONObject, *args: Any, **kwargs: Any) -> JSONObject:
        if payload["metadata"]["component"] == "generator":
            scales.extend(json.loads(_message(message))["scale"]
                          for message in payload["input"] if message["role"] == "user")
        return original_call(api, endpoint, payload, *args, **kwargs)

    monkeypatch.setattr(ExperimentAPI, "call", record)
    assert cli("run", config, data, manifest, *args, "--query", str(query),
               "--output", str(tmp_path / "result.json")) == 0
    assert scales == [0.5] * 4 + [0.7]


class ByteEncoding:
    def encode(self, text: str, **_kwargs: object) -> bytes:
        return text.encode("utf-8")


def test_full_context_limit_blocks_dispatch_and_changes_to_prompt_change_fingerprint(
    tmp_path: Path, pilot: PilotData, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(cache_dir=str(tmp_path / "cache"), generator_tokens=128)
    api = create_inference_api(config, "mock", encoding=ByteEncoding())
    old = create_predicate_components(config, api, context_limit=4096, scale_arm="unscaled")
    changed = create_predicate_components(
        config, api, context_limit=4096, generator_prompt="More detail. " * 1024, scale_arm="unscaled",
    )
    support_question = {
        "observation": pilot.training[0]["start_observation"], "task_goal": TASK_GOAL * 200, "scale": None,
    }
    too_many_tokens = create_predicate_components(
        config, api, context_limit=4096,
        generator_examples=[(support_question, PROPOSAL)], scale_arm="unscaled",
    )
    assert old.contract("generator")["fingerprint"] != changed.contract("generator")["fingerprint"]
    assert old.contract("policy")["fingerprint"] == changed.contract("policy")["fingerprint"]
    assert old.contract("generator")["examples_sha256"] != too_many_tokens.contract("generator")["examples_sha256"]
    observation = SokobanObservation(tuple(pilot.held_out["start_observation"]))

    def unexpected_call(*_args: Any, **_kwargs: Any) -> JSONObject:
        pytest.fail("Over-context request reached the transport")

    monkeypatch.setattr(api, "call", unexpected_call)
    with pytest.raises(BudgetExceeded, match="context window"):
        changed.propose(observation, TASK_GOAL, None, mock=lambda: json.dumps(PROPOSAL))
    with pytest.raises(BudgetExceeded, match="context window"):
        too_many_tokens.propose(observation, TASK_GOAL, None, mock=lambda: json.dumps(PROPOSAL))
    assert api.metrics["logical_requests"] == api.metrics["physical_requests"] == 0


@pytest.mark.parametrize("row,col", [(1, 2), (0, 1)])
def test_generator_rejects_satisfied_or_illegal_intents(tmp_path: Path, pilot: PilotData, row: int, col: int) -> None:
    config = ExperimentConfig(cache_dir=str(tmp_path / "cache"), generator_tokens=128)
    api = create_inference_api(config, "mock", encoding=ByteEncoding())
    components = create_predicate_components(config, api, context_limit=8192, scale_arm="unscaled")
    observation = SokobanObservation(tuple(pilot.held_out["start_observation"]))
    output = {"effects": [{"predicate": "box_at", "row": row, "col": col}], "preserve": [], "instruction": None}
    with pytest.raises(ValueError):
        components.propose(observation, TASK_GOAL, None, mock=lambda: json.dumps(output))
    assert api.metrics["logical_requests"] == 1
    assert api.metrics["physical_requests"] == 0
