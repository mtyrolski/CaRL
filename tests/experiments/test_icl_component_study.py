"""Component study uses real witnesses while its tests remain entirely offline."""
import json
from pathlib import Path

import pytest

from carl.icl.api import CachedOpenAI
from carl.icl.boards import from_rows
from carl.icl.evaluation.component_study import export, figures, load_cases, summarize
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.datasets import environment, load_data
from scripts.evaluate_component_study import experiment_config, load_study, plan, run

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "configs/icl/component_study/luna_100.json"


@pytest.fixture
def prepared_data() -> None:
    if not (ROOT / "data/icl/component_study_100_clean/real_12x12x4/test.json").exists():
        pytest.skip("Prepare the 100-trajectory holdout to run this integration test")


def test_zero_shot_and_few_shot_counts_are_exact(tmp_path: Path, prepared_data: None) -> None:
    study = load_study(STUDY)
    study.instances = 1
    for count in study.few_shot:
        config = experiment_config(study, count)
        training, instances = load_data(config)
        board = from_rows(instances[0]["state"])
        components = LunaComponents(config, ExperimentAPI(tmp_path, "mock", config.limits), environment(board), training)
        for payload in (components.policy_payload(board, board), components.generator_payload(board, 4),
                        components.value_payload(board)):
            messages = payload["input"]
            assert sum(message["role"] == "assistant" for message in messages) == count
            assert messages[-1]["role"] == "user"
    assert plan(study)["max_requests_all_variants"] == 48


def test_plots_count_errors_and_constant_value_correctly() -> None:
    manifest = {"study": {"instances": 2, "few_shot": [0], "value_positions": [0, 0.5, 1],
                           "monotonic_tolerance": 1e-6}}
    base = {"few_shot": 0}
    cases = [
        {**base, "component": "policy", "reached": True},
        {**base, "component": "policy", "reached": False, "error": "api"},
        {**base, "component": "generator", "generator": {"valid_count": 0}},
        {**base, "component": "generator", "generator": {"valid_count": 1}, "rollout": {"reached": True}},
        {**base, "component": "value", "values": [{"fraction": i, "value": 0.5} for i in [0, 0.5, 1]]},
    ]
    stats = {row["metric"]: row for row in summarize(manifest, cases)}
    assert stats["policy_reached"]["observed"] == 1
    assert stats["policy_reached"]["planned"] == 2
    assert stats["generated_reached_all"]["rate_pct"] == 50
    assert stats["generated_reached_given_valid"]["rate_pct"] == 100
    assert stats["value_mean_nondecreasing_edges"]["rate_pct"] == 100
    assert stats["value_mean_increasing_edges"]["rate_pct"] == 0
    assert stats["value_mean_tied_edges"]["rate_pct"] == 100
    assert len(figures(manifest, cases)) == 4


def test_mock_run_resume_and_plot_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared_data: None) -> None:
    import plotly.graph_objects as go

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Offline study must never create a live API client")

    monkeypatch.setattr(CachedOpenAI, "_client", forbidden)
    study = load_study(STUDY)
    study.instances = 2
    study.few_shot = [0, 2]
    study.output_dir = str(tmp_path / "results")
    study.cache_dir = str(tmp_path / "cache")
    selected = ("policy", "generator", "value")
    output = run(study, "mock", selected=selected, shots=(0, 2), resume=False, retry_errors=False)
    manifest, cases = load_cases(output)
    assert len(cases) == 12
    assert all("error" not in case for case in cases)
    assert all(len(case["values"]) == 5 for case in cases if case["component"] == "value")
    before = {path: path.read_bytes() for path in (output / "cases").rglob("*.json")}
    run(study, "mock", selected=selected, shots=(0, 2), resume=True, retry_errors=False)
    assert before == {path: path.read_bytes() for path in before}
    plot_dir = export(output)
    assert len(list(plot_dir.glob("*.html"))) == 4
    assert (plot_dir / "summary.csv").exists() and (plot_dir / "usage.csv").exists()
    assert manifest["provider"] == "mock"
    config_path = tmp_path / "study.json"
    config_path.write_text(study.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)
    notebook = json.loads((ROOT / "notebooks/luna_component_study.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]).replace(
                'ROOT / "configs/icl/component_study/luna_100.json"', f"Path({str(config_path)!r})"
            )
            exec(compile(source, "component_study_notebook", "exec"), namespace)
    assert len(namespace["cases"]) == 12
    study.horizon = 3
    with pytest.raises(ValueError, match="manifest differs"):
        run(study, "mock", selected=selected, shots=(0, 2), resume=True, retry_errors=False)


def test_plan_is_real_luna_and_prepared_holdout_has_100_disjoint_cases(prepared_data: None) -> None:
    study = load_study(STUDY)
    assert study.model == "gpt-6-luna" and study.instances == 100 and study.few_shot == [0, 2, 4]
    config = experiment_config(study, 4)
    training, instances = load_data(config)
    assert len(instances) == 100 and len(training["contexts"]) >= 4
    witnesses = json.loads((config.path("data_dir") / "private_witnesses" / config.family / "test.json").read_text())
    assert {item["id"] for item in instances} == {item["id"] for item in witnesses}
    selected = {item["level_key"] for item in instances}
    for name in ("fresh_holdout_20261102", "fresh_verified_holdout_20261102", "final_simple_holdout_20261102"):
        previous = ROOT / "data/icl" / name / config.family / "test.json"
        if previous.exists():
            assert not selected.intersection(item["level_key"] for item in json.loads(previous.read_text())["instances"])


def test_openai_sdk_streams_are_exercised_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        prepared_data: None) -> None:
    import httpx
    import openai

    from scripts import evaluate_component_study as runner

    captured = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        component = payload["metadata"]["component"]
        answer = {"policy": {"probabilities": [0.25, 0.25, 0.25, 0.25]},
                  "generator": {"subgoals": []}, "value": {"remaining_actions": 10}}[component]
        response = {"id": "resp_test", "object": "response", "created_at": 0, "model": "gpt-6-luna",
                    "status": "completed", "output": [{"id": "msg_test", "type": "message",
                    "role": "assistant", "status": "completed", "content": [{"type": "output_text",
                    "text": json.dumps(answer), "annotations": []}]}], "usage": {"input_tokens": 10,
                    "output_tokens": 5, "total_tokens": 15}}
        event = {"type": "response.completed", "sequence_number": 0, "response": response}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text=f"event: response.completed\ndata: {json.dumps(event)}\n\n")

    transport = httpx.Client(transport=httpx.MockTransport(handle))
    client = openai.OpenAI(api_key="test-only", http_client=transport, max_retries=0)
    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: client)
    monkeypatch.setattr(runner, "credentials", lambda _provider: None)
    study = load_study(STUDY)
    study.instances = 1
    study.few_shot = [0]
    study.output_dir = str(tmp_path / "results")
    study.cache_dir = str(tmp_path / "cache")
    output = run(study, "openai", selected=("policy", "generator", "value"), shots=(0,),
                 resume=False, retry_errors=False)
    _manifest, cases = load_cases(output)
    assert len(cases) == 3 and all("error" not in case for case in cases)
    assert {payload["metadata"]["component"] for payload in captured} == {"policy", "generator", "value"}
    assert all(payload["model"] == "gpt-6-luna" and payload["stream"] for payload in captured)
    assert all(not any(message["role"] == "assistant" for message in payload["input"])
               for payload in captured if payload["metadata"]["component"] != "generator")


def test_retry_errors_preserves_previous_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               prepared_data: None) -> None:
    from scripts import evaluate_component_study as runner

    study = load_study(STUDY)
    study.instances = 1
    study.few_shot = [0]
    study.output_dir = str(tmp_path / "results")
    study.cache_dir = str(tmp_path / "cache")
    calls = []

    def case(*_args: object, **_kwargs: object) -> dict:
        calls.append(1)
        result = {"component": "policy", "few_shot": 0, "metrics": {"requests": 1}, "reached": False}
        return {**result, "error": "test failure"} if len(calls) == 1 else result

    monkeypatch.setattr(runner, "case_result", case)
    output = run(study, "mock", selected=("policy",), shots=(0,), resume=False, retry_errors=False)
    run(study, "mock", selected=("policy",), shots=(0,), resume=True, retry_errors=False)
    assert len(calls) == 1
    run(study, "mock", selected=("policy",), shots=(0,), resume=True, retry_errors=True)
    _manifest, cases = load_cases(output)
    assert len(calls) == 2 and "error" not in cases[0]
    assert cases[0]["prior_attempts"][0]["metrics"]["requests"] == 1
