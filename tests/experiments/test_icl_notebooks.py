"""Notebook startup and component walkthrough must work without provider access."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ("luna_components_demo.ipynb", "luna_icl_evaluation.ipynb", "luna_trajectory_diagnostics.ipynb")


def test_notebook_typecheck_maps_generated_lines_to_cells(tmp_path):
    from scripts.check_notebook_types import extract_notebook

    notebook = tmp_path / "example.ipynb"
    notebook.write_text(json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["Notes"]},
        {"cell_type": "code", "source": ["value = 1\n", "print(value)"]},
        {"cell_type": "code", "source": "result = value"},
    ]}))
    extracted = tmp_path / "example.py"
    locations = extract_notebook(notebook, extracted)
    assert extracted.read_text().splitlines() == [
        "# Notebook cell 1", "value = 1", "print(value)", "",
        "# Notebook cell 2", "result = value", "",
    ]
    assert locations == {2: (1, 1), 3: (1, 2), 6: (2, 1)}


def code_cells(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


@pytest.mark.parametrize("name", NOTEBOOKS)
@pytest.mark.parametrize("location", (ROOT.parent, ROOT, ROOT / "notebooks"))
def test_notebook_finds_checkout(name, location, monkeypatch):
    monkeypatch.chdir(location)
    # Do not let the current interpreter conceal a broken workspace lookup.
    monkeypatch.setattr(sys, "executable", "/unrelated/bin/python")
    namespace = {}
    bootstrap = code_cells(name)[0].split("print(f'CaRL checkout:")[0]
    exec(bootstrap, namespace)
    assert namespace["ROOT"] == ROOT


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_missing_checkout_is_actionable(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "executable", "/unrelated/bin/python")
    bootstrap = code_cells(name)[0].split("print(f'CaRL checkout:")[0]
    with pytest.raises(RuntimeError, match="Set REPO_PATH"):
        exec(bootstrap, {})


@pytest.mark.parametrize(
    "family,size,boxes,run_search",
    [
        ("real_12x12x4", 12, 4, False),
        ("real_12x12x4", 12, 4, True),
        ("generated_32x32x10", 32, 10, False),
    ],
)
def test_component_notebook_runs_offline_from_workspace(
    tmp_path, monkeypatch, family, size, boxes, run_search
):
    import numpy as np
    import plotly.graph_objects as go

    from carl.icl.api import CachedOpenAI
    from carl.icl.boards import from_rows, level_key, tiles
    from carl.icl.datasets import environment
    from carl.icl import experiment

    if not (ROOT / "data/icl" / family / "test.json").exists():
        pytest.skip("Run make icl-prepare to verify notebooks against prepared data")
    if not (ROOT / "data/icl/value_examples.json").exists():
        pytest.skip("Run prepare-value for the tuned notebook profile")

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Offline notebook must not load credentials or create an API client"
        )

    figures = []
    monkeypatch.setattr(CachedOpenAI, "_client", forbidden)
    monkeypatch.setattr(experiment, "load_dotenv", forbidden)
    monkeypatch.setattr(
        go.Figure, "show", lambda self, **_: figures.append(self.to_json())
    )
    monkeypatch.chdir(ROOT.parent)
    namespace = {}
    for index, source in enumerate(code_cells("luna_components_demo.ipynb")):
        if run_search:
            source = source.replace("RUN_SEARCH = False", "RUN_SEARCH = True")
        if index == 0:
            source = source.replace(
                "PROVIDER: Provider = 'openai'", "PROVIDER: Provider = 'mock'"
            )
            source = source.replace(
                "FAMILY: Final = 'real_12x12x4'", f"FAMILY: Final = '{family}'"
            )
        exec(compile(source, f"notebook_cell_{index}", "exec"), namespace)
        if index == 0:
            namespace["config"].cache_dir = str(tmp_path / "cache")
    assert namespace["api"].metrics["logical_requests"] <= 7
    assert namespace["api"].metrics["generator_calls"] == 1
    assert namespace["api"].metrics["policy_calls"] >= 1
    assert namespace["api"].metrics["value_calls"] >= 2
    assert namespace["history"]
    if run_search:
        assert namespace["outcome"].attempted_actions <= namespace["SOLVER_ACTION_BUDGET"]
    assert namespace["tree"] is not None
    assert len(namespace["tree"].expansions) == namespace["outcome"].expansions
    if family == "real_12x12x4":
        assert namespace["fixture_outcome"].solved
        assert namespace["fixture_replay"]
    assert len(figures) >= 4
    initial = namespace["initial"]
    assert initial.shape == (size, size, 7)
    assert np.isin(tiles(initial), [3, 4]).sum() == boxes
    assert not namespace["config"].easy
    assert namespace["config"].few_shot == 4
    assert level_key(initial) != level_key(namespace["demo"])
    for example in namespace["examples"]:
        current = from_rows(example["state"])
        example_env = environment(current)
        for action in example["actions"]:
            current = np.asarray(example_env.next_state(current, action))
        np.testing.assert_array_equal(current, from_rows(example["subgoal"]))
        assert len(example["actions"]) == example["k"]
    for (question, answer), context in zip(
        namespace["policy_examples"], namespace["training"]["contexts"], strict=False
    ):
        candidate = next(
            item
            for item in context["candidates"]
            if item["subgoal"] == question["subgoal"]
        )
        assert int(np.argmax(answer["probabilities"])) == candidate["actions"][0]


def test_component_notebook_does_not_invent_target_when_generator_fails(tmp_path, monkeypatch):
    import plotly.graph_objects as go

    from carl.icl.luna import LunaComponents

    if not (ROOT / "data/icl/real_12x12x4/test.json").exists():
        pytest.skip("Run make icl-prepare to verify the prepared notebook data")
    if not (ROOT / "data/icl/value_examples.json").exists():
        pytest.skip("Run prepare-value for the tuned notebook profile")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(LunaComponents, "propose", lambda *_args: [])
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)
    namespace = {}
    for index, source in enumerate(code_cells("luna_components_demo.ipynb")):
        exec(compile(source, f"notebook_cell_{index}", "exec"), namespace)
        if index == 0:
            namespace["config"].cache_dir = str(tmp_path / "cache")
        if "Generator diagnostics:" in source:
            break
    assert namespace["proposals"] == []
    assert namespace["target"] is None


@pytest.fixture
def campaign_csv_tables(tmp_path):
    """Small deterministic export contract: 30 cells, 22 completed independent runs."""
    import gzip

    import pandas as pd

    directory = tmp_path / "plot_data"
    directory.mkdir()

    def table(name, rows):
        pd.DataFrame(rows).to_csv(directory / f"{name}.csv", index=False)

    runs, rates, pairs = [], [], []
    for budget, observed, relative_hits, absolute_hits in ((128, 5, 2, 1), (256, 4, 1, 3), (512, 2, 2, 0)):
        for method, hits in (("relative_u", relative_hits), ("absolute_d", absolute_hits)):
            for board in range(5):
                complete = board < observed
                runs.append({"run_key": f"{budget}:{method}:{board}", "board_id": f"board-{board}",
                             "family": "real_12x12x4", "method": method, "budget_actions": budget,
                             "execution_status": "complete" if complete else "missing",
                             "solved": board < hits if complete else None,
                             "attempted_actions": budget - 1 if complete else None,
                             "input_tokens": 100 if complete else None, "output_tokens": 10 if complete else None,
                             "provider_requests": 2 if complete else None,
                             "completion_class": ("solved" if board < hits else "budget_exhausted") if complete else None,
                             "cost_settled_usd": 0.001 if complete else 0, "cost_unknown_reserved_usd": 0})
            rates.append({"family": "real_12x12x4", "method": method, "budget_actions": budget,
                          "verified_successes": hits, "denominator_boards": observed, "expected_boards": 5,
                          "solve_rate": hits / observed, "wilson_low": 0, "wilson_high": 1})
        pairs.append({"family": "real_12x12x4", "budget_actions": budget, "complete_pairs": observed,
                      "expected_pairs": 5, "delta_relative_minus_absolute": (relative_hits - absolute_hits) / observed,
                      "full_cohort_delta_lower": -1, "full_cohort_delta_upper": 1})
    table("runs", runs)
    table("solve_rate_by_budget", rates)
    table("paired_deltas", pairs)
    (directory / "validation.json").write_text(json.dumps({"completed_runs": 22, "expected_runs": 30}))
    board = ["#####", "#@$.#", "#####"]
    target = ["#####", "#-@*#", "#####"]
    table("figure1_targets", [{"context_id": "training-1", "target_id": f"target-{k}", "k": k,
                                "state": json.dumps(board), "p_hat": 1, "R_attempted": 2, "hits": 2}
                               for k in (1, 2)])
    table("figure1_ecdf", [{"context_id": "training-1", "d": k, "F_right": k / 2} for k in (1, 2)])
    table("figure1_quantiles", [{"context_id": "training-1", "percentile": 0.5, "quantile_d": 1}])
    table("reachability_by_offset", [{"k": 1, "mean_target_p_hat": 1}])
    table("component_policy", [{"variant": "frozen", "k": 1, "exact_hit": hit} for hit in (True, False)])
    table("component_generator", [{"variant": "frozen", "k": 1, "reached_any": hit} for hit in (True, False)])
    table("component_value", [{"variant": "current_frozen", "board_id": "validation-1", "position": 0,
                               "remaining": 2, "predicted_value": 0.8, "reference_value": 0.9}])
    table("component_costs", [{"stage": "fixture", "provider_attempts": 44, "cost_settled_usd": 0.022,
                               "cost_unknown_reserved_usd": 0}])
    comparison = [{"board_id": f"validation-{index}", "method": method, "budget_actions": 128,
                   "solved": method == "torch", "attempted_actions": 1, "solution_length": 1 if method == "torch" else None,
                   "expansions": 1, "policy_mode": "fixture", "replay_verified": True}
                  for index in (1, 2) for method in ("torch", "relative_u", "absolute_d")]
    table("torch_luna_validation_runs", comparison)
    table("torch_luna_validation_summary", [{"method": "torch", "solved": 2, "observed": 2}])
    with gzip.open(directory / "search_attempts.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"method": "relative_u", "scale": 0.5, "source": board, "proposal": target,
                                 "actual_endpoint": board, "reached": False}) + "\n")
    return directory


@pytest.fixture
def offline_evaluation(monkeypatch):
    """Execute real CSV plotting code while denying credentials, API and sockets."""
    import socket

    import plotly.graph_objects as go

    from carl.icl import experiment
    from carl.icl.api import CachedOpenAI

    def forbidden(*_args, **_kwargs):
        pytest.fail("Offline evaluation attempted credentials, API or network access")

    monkeypatch.setattr(CachedOpenAI, "_client", forbidden)
    monkeypatch.setattr(experiment, "load_dotenv", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    figures = []
    monkeypatch.setattr(go.Figure, "show", lambda self, **_: figures.append(self))
    monkeypatch.chdir(ROOT.parent)

    def execute(directory):
        namespace = {"DATA_PATH": directory}
        for index, source in enumerate(code_cells("luna_icl_evaluation.ipynb")):
            exec(compile(source, f"evaluation_cell_{index}", "exec"), namespace)
        return namespace, figures

    return execute


def test_evaluation_notebook_runs_offline(offline_evaluation, campaign_csv_tables):
    namespace, figures = offline_evaluation(campaign_csv_tables)
    runs = namespace["runs"]
    assert namespace["DATA"] == campaign_csv_tables
    assert len(runs) == 30 and runs.run_key.is_unique
    assert runs.execution_status.eq("complete").sum() == 22
    assert runs.loc[runs.execution_status != "complete", "solved"].isna().all()
    assert set(namespace["figures"]) == {"A_solve_rate", "B_calibration", "C_components", "D_cost_stops"}
    assert len(figures) == 5
    assert namespace["attempt"]["actual_endpoint"] != namespace["attempt"]["proposal"]
    assert not namespace["attempt"]["reached"]
    with pytest.raises(RuntimeError, match="External network disabled"):
        namespace["deny_external_connections"]("socket.connect", (None, ("example.org", 443)))


def test_evaluation_notebook_shows_independently_run_budgets(offline_evaluation, campaign_csv_tables):
    namespace, _ = offline_evaluation(campaign_csv_tables)
    figure = namespace["figures"]["A_solve_rate"]
    relative = next(trace for trace in figure.data if trace.name == "relative_u")
    absolute = next(trace for trace in figure.data if trace.name == "absolute_d")
    # B=512 successes take 511 actions. Truncating them would invent 0% at B=128/256,
    # whereas the independently executed fixture runs have their own measured rates.
    assert list(relative.x) == [128, 256, 512]
    assert list(relative.y) == [40, 25, 100]
    assert list(absolute.y) == [20, 75, 0]
    assert [int(row[0]) for row in relative.customdata] == [5, 4, 2]
    assert [int(row[1]) for row in relative.customdata] == [5, 5, 5]
    assert relative.mode == "markers"


def test_evaluation_notebook_omits_missing_budget_points(offline_evaluation, campaign_csv_tables):
    import pandas as pd

    path = campaign_csv_tables / "solve_rate_by_budget.csv"
    rates = pd.read_csv(path)
    selected = rates.budget_actions.eq(512)
    rates.loc[selected, ["solve_rate", "wilson_low", "wilson_high"]] = float("nan")
    rates.loc[selected, ["denominator_boards", "verified_successes"]] = 0
    rates.to_csv(path, index=False)
    runs_path = campaign_csv_tables / "runs.csv"
    runs = pd.read_csv(runs_path)
    selected = runs.budget_actions.eq(512)
    runs.loc[selected, "execution_status"] = "missing"
    runs.loc[selected, ["solved", "attempted_actions", "input_tokens", "output_tokens",
                        "provider_requests", "completion_class"]] = None
    runs.loc[selected, "cost_settled_usd"] = 0
    runs.to_csv(runs_path, index=False)
    pair_path = campaign_csv_tables / "paired_deltas.csv"
    pairs = pd.read_csv(pair_path)
    selected = pairs.budget_actions.eq(512)
    pairs.loc[selected, "complete_pairs"] = 0
    pairs.loc[selected, "delta_relative_minus_absolute"] = float("nan")
    pairs.to_csv(pair_path, index=False)
    namespace, _ = offline_evaluation(campaign_csv_tables)
    traces = namespace["figures"]["A_solve_rate"].data
    assert all(list(trace.x) == [128, 256] for trace in traces if trace.name in ("relative_u", "absolute_d"))


@pytest.mark.parametrize("via_environment", [False, True])
def test_evaluation_notebook_supports_checkout_override(tmp_path, monkeypatch, via_environment):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "executable", "/unrelated/bin/python")
    namespace = {} if via_environment else {"REPO_PATH": ROOT}
    if via_environment:
        monkeypatch.setenv("REPO_PATH", str(ROOT))
    exec(code_cells("luna_icl_evaluation.ipynb")[0], namespace)
    assert namespace["ROOT"] == ROOT


def test_evaluation_notebook_rejects_invalid_checkout_override(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    with pytest.raises(RuntimeError, match="Set REPO_PATH"):
        exec(code_cells("luna_icl_evaluation.ipynb")[0], {"REPO_PATH": tmp_path})


def test_demo_notebook_live_sdk_with_fake_streaming_transport(tmp_path, monkeypatch):
    """Exercise the real SDK, all roles and simulator with HTTP isolated locally."""
    import httpx
    import numpy as np
    import plotly.graph_objects as go
    from openai import OpenAI

    from carl.icl import experiment
    from carl.icl.boards import from_rows, to_rows
    from carl.icl.datasets import environment

    if not (ROOT / "data/icl/real_12x12x4/test.json").exists():
        pytest.skip("Prepared dataset required")
    if not (ROOT / "data/icl/value_examples.json").exists():
        pytest.skip("Run prepare-value for the tuned notebook profile")

    captured = []

    def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        role = payload["metadata"]["component"]
        query = json.loads(payload["input"][-1]["content"][0]["text"])
        state = from_rows(query["state"])
        if role == "generator":
            env = environment(state)
            target = next(
                np.asarray(env.next_state(state, action))
                for action in range(4)
                if not np.array_equal(env.next_state(state, action), state)
            )
            rows = to_rows(target)
            answer = {"subgoals": [{f"row_{index:02d}": row for index, row in enumerate(rows)}]}
        elif role == "policy":
            env = environment(state)
            target = from_rows(query["subgoal"])
            action = next(
                action
                for action in range(4)
                if np.array_equal(env.next_state(state, action), target)
            )
            probabilities = [0.0] * 4
            probabilities[action] = 1.0
            answer = {"probabilities": probabilities}
        else:
            answer = {"remaining_actions": 20}
        text = json.dumps(answer)
        response = {
            "id": "resp_local",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6-luna",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_local",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        delta = {
            "type": "response.output_text.delta",
            "sequence_number": 0,
            "item_id": "msg_local",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        }
        complete = {
            "type": "response.completed",
            "sequence_number": 1,
            "response": response,
        }
        body = f"event: {delta['type']}\ndata: {json.dumps(delta)}\n\nevent: {complete['type']}\ndata: {json.dumps(complete)}\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=body
        )

    monkeypatch.setattr(experiment, "credentials", lambda _provider: None)
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)
    monkeypatch.chdir(ROOT.parent)
    namespace = {}
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = OpenAI(
            api_key="local-fixture-not-a-secret", http_client=transport, max_retries=0
        )
        for index, source in enumerate(code_cells("luna_components_demo.ipynb")):
            if index == 0:
                source = source.replace(
                    "PROVIDER: Provider = 'mock'", "PROVIDER: Provider = 'openai'"
                )
            exec(compile(source, f"sdk_notebook_{index}", "exec"), namespace)
            if index == 0:
                namespace["config"].cache_dir = str(tmp_path / "cache")
            if source.startswith("components = components_for_run"):
                namespace["api"].client = client
    assert {item["metadata"]["component"] for item in captured} == {
        "generator",
        "policy",
        "value",
    }
    assert all(
        item["model"] == "gpt-6-luna" and item["stream"] and not item["store"]
        for item in captured
    )
    assert all(
        sum(m["role"] == "assistant" for m in item["input"])
        == (16 if item["metadata"]["component"] == "value" else 4)
        for item in captured
    )
    assert namespace["attempt"].reached and len(namespace["history"]) == 1
    assert namespace["api"].metrics["requests"] >= 3


def test_demo_uses_prompt_overrides_from_child_config(tmp_path, monkeypatch):
    import plotly.graph_objects as go
    from carl.icl.api import CachedOpenAI

    if not (ROOT / "data/icl/real_12x12x4/test.json").exists():
        pytest.skip("Prepared dataset required")
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps(
            {
                "extends": str(ROOT / "configs/icl/budget_sweeps/quick_fixed.json"),
                "prompts": {
                    "policy": "Configured policy text",
                    "generator": "Configured generator text",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        CachedOpenAI, "_client", lambda *_: pytest.fail("Unexpected API client")
    )
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)
    monkeypatch.chdir(ROOT.parent)
    namespace = {}
    for index, source in enumerate(code_cells("luna_components_demo.ipynb")):
        if index == 0:
            source = source.replace(
                "'configs/icl/experiments/tuned_fixed.json'", repr(str(custom))
            )
        exec(compile(source, f"custom_prompt_cell_{index}", "exec"), namespace)
        if source.startswith("components = components_for_run"):
            break
    component = namespace["components"]
    board = namespace["initial"]
    generator = component.generator_payload(board, namespace["scale"])
    policy = component.policy_payload(board, board)
    assert generator["input"][0]["content"][0]["text"].startswith(
        "Configured generator text"
    )
    assert policy["input"][0]["content"][0]["text"].startswith("Configured policy text")
