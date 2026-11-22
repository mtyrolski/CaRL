from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.icl.api import BudgetExceeded, CachedOpenAI, RequestFailed, write_json
from carl.icl.boards import from_rows, level_key, to_rows, validate_subgoal
from carl.icl.generator import OpenAISubgoalGenerator
from carl.icl.types import Board
from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy
from carl.inference_components.subgoal_generator import AdaptiveSubgoalGenerator, SubgoalGenerator
from carl.inference_components.validator import BasicValidator
from carl.inference_components.value import Value
from carl.planners.adasubs import AdasubsPlanner, UniversalPropositionalPlanner
from carl.planners.base import Planner
from carl.solver.nodes import SearchTreeNode
from carl.solver.subgoal_search import Solver
from carl.utils.aliases import State

START = ["#####", "#---#", "#@$.#", "#---#", "#####"]
TARGET = ["#####", "#---#", "#-@*#", "#---#", "#####"]
DEMO = ["#####", "#-#-#", "#@$.#", "#---#", "#####"]
DEMO_TARGET = ["#####", "#-#-#", "#-@*#", "#---#", "#####"]


def response(text, status="completed"):
    return SimpleNamespace(status=status, output_text=text, output=[], model="test-model", id="test-response",
        usage=SimpleNamespace(model_dump=lambda: dict(input_tokens=10, output_tokens=5,
            input_tokens_details=dict(cached_tokens=3))))


@pytest.fixture
def demonstration_file(tmp_path):
    path = tmp_path / "demonstrations.json"
    write_json(path, dict(demonstrations=[dict(state=DEMO, subgoal=DEMO_TARGET, k=1)]))
    return path


def generator(tmp_path, demonstration_file, *, client=None, provider="openai", **kwargs):
    api = CachedOpenAI(tmp_path / "cache", provider=provider, client=client)
    return OpenAISubgoalGenerator(env=None, model="test-model", demonstrations_path=demonstration_file,
                                 api=api, generator_k_list=[1], **kwargs)


def test_board_roundtrip_and_invariants():
    state, target = from_rows(START), from_rows(TARGET)
    assert state.shape == (5, 5, 7)
    assert to_rows(state) == START
    validate_subgoal(state, target)
    assert level_key(state) == level_key(target)
    assert level_key(state) != level_key(from_rows(DEMO))
    for bad in (["??"], ["###", "##"], [], "not rows"):
        with pytest.raises(ValueError):
            cast(Callable[[object], Board], from_rows)(bad)
    for bad in (["#####", "#-#-#", "#-@*#", "#---#", "#####"],
                ["#####", "#---#", "#--*#", "#---#", "#####"],
                ["#####", "#---#", "#-@.#", "#---#", "#####"]):
        with pytest.raises(ValueError):
            validate_subgoal(state, from_rows(bad))


@pytest.mark.parametrize("planner,k", [("adaptive", 1), ("universal", None)])
def test_prompt_contract(tmp_path, demonstration_file, planner, k):
    gen = generator(tmp_path, demonstration_file, planner=planner)
    payload = gen.build_payload(SearchTreeNode(from_rows(START), 0, None, None, k))
    query = json.loads(payload["input"][-1]["content"])
    assert query["state"] == START
    assert set(query) == ({"state", "k", "num_subgoals", "controller_cap"} if k else {"state", "num_subgoals", "controller_cap"})
    assert payload["store"] is False
    assert payload["text"]["format"]["strict"] is True
    assert TARGET not in [json.loads(message["content"]) for message in payload["input"][1:]]
    with pytest.raises(ValueError):
        gen.build_payload(SearchTreeNode(from_rows(START), 0, None, None, None if k else 1))


def test_filters_self_copies_duplicates_and_changed_layout(tmp_path, demonstration_file):
    output = json.dumps(dict(subgoals=[dict(rows=START), dict(rows=TARGET), dict(rows=TARGET), dict(rows=DEMO_TARGET)]))
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response(output)))
    gen = generator(tmp_path, demonstration_file, client=client, num_subgoals=4)
    proposals = gen.get_subgoals(SearchTreeNode(from_rows(START), 0, None, None, 1))
    assert len(proposals) == 1
    np.testing.assert_array_equal(proposals[0].state, from_rows(TARGET))
    assert gen.metrics == dict(expansions=1, decoded=4, invalid=1, self_copies=1, duplicates=1, malformed=0)


def test_api_cache_budget_and_offline_replay(tmp_path):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response('{"subgoals": []}')
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    api = CachedOpenAI(tmp_path, provider="openai", max_requests=1, client=client)
    payload = dict(model="test", input="board")
    original = api.call("responses", payload)
    assert api.call("responses", payload) == original
    assert len(calls) == 1 and api.metrics["cache_hits"] == 1
    assert api.metrics["input_tokens"] == 10 and api.metrics["cached_input_tokens"] == 3
    with pytest.raises(BudgetExceeded):
        api.call("responses", dict(payload, input="other"))
    replay = CachedOpenAI(tmp_path, provider="replay")
    assert replay.call("responses", payload) == original
    assert replay.client is None and replay.metrics["requests"] == 0
    with pytest.raises(RequestFailed, match="No completed cached"):
        replay.call("responses", dict(payload, model="different"))


def test_mock_cache_cannot_satisfy_live_replay(tmp_path):
    mock = CachedOpenAI(tmp_path, provider="mock")
    mock.call("responses", {}, mock=lambda: dict(output_text="mock", usage={}))
    with pytest.raises(RequestFailed):
        CachedOpenAI(tmp_path, provider="replay").call("responses", {})


def test_errors_do_not_cache_credentials_or_count_as_completed(tmp_path):
    def fail(**kwargs):
        raise RuntimeError("secret-key-must-not-be-recorded")
    api = CachedOpenAI(tmp_path, provider="openai", client=SimpleNamespace(responses=SimpleNamespace(create=fail)))
    with pytest.raises(RequestFailed) as error:
        api.call("responses", {})
    assert "secret-key" not in str(error.value)
    assert "secret-key" not in next((tmp_path / "openai").glob("*.json")).read_text()
    with pytest.raises(RequestFailed):
        CachedOpenAI(tmp_path, provider="replay").call("responses", {})
    assert api.metrics["requests"] == 1


@pytest.mark.parametrize("output,status", [("{", "completed"), ('{"subgoals": "wrong"}', "completed"),
                                             ('{"subgoals": []}', "incomplete")])
def test_malformed_and_incomplete_results(tmp_path, demonstration_file, output, status):
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response(output, status)))
    gen = generator(tmp_path, demonstration_file, client=client)
    node = SearchTreeNode(from_rows(START), 0, None, None, 1)
    if status == "incomplete":
        with pytest.raises(RequestFailed):
            gen.get_subgoals(node)
    else:
        assert gen.get_subgoals(node) == []
        assert gen.metrics["malformed"] == 1


def test_embedding_selection_is_cached_and_keeps_horizon(tmp_path, demonstration_file):
    second = ["#####", "##--#", "#@$.#", "#---#", "#####"]
    second_target = ["#####", "##--#", "#-@*#", "#---#", "#####"]
    write_json(demonstration_file, dict(demonstrations=[dict(state=DEMO, subgoal=DEMO_TARGET, k=1),
        dict(state=second, subgoal=second_target, k=1)]))
    def embed(**kwargs):
        vector = [1., 0.] if "##--#" in kwargs["input"][0] or "#---#\n#@$.#" in kwargs["input"][0] else [0., 1.]
        return SimpleNamespace(model="embedding-test", data=[SimpleNamespace(index=0, embedding=vector)],
                               usage=SimpleNamespace(model_dump=lambda: dict(prompt_tokens=7)))
    gen = generator(tmp_path, demonstration_file, selection="embeddings", examples_per_request=1,
                    client=SimpleNamespace(embeddings=SimpleNamespace(create=embed)))
    node = SearchTreeNode(from_rows(START), 0, None, None, 1)
    payload = gen.build_payload(node)
    assert json.loads(payload["input"][1]["content"])["state"] == second
    assert gen.api.metrics["embedding_requests"] == 3
    gen.build_payload(node)
    assert gen.api.metrics["embedding_requests"] == 3 and gen.api.metrics["cache_hits"] == 3
    before = dict(gen.api.metrics)
    gen.build_payload(node, resolve_embeddings=False)
    assert gen.api.metrics == before


@pytest.mark.parametrize("planner", ["adaptive", "universal"])
def test_full_existing_solver_replays_solution_with_icl_adapter(tmp_path, demonstration_file, planner):
    env = SokobanEnv(SokobanTokenizer(size_of_board=(5, 5)), num_boxes=1)
    gen = generator(tmp_path, demonstration_file, provider="mock", planner=planner, controller_cap=1)
    gen.env = env
    class RightPolicy:
        def construct_network(self):
            pass
        def get_action(self, state, subgoal):
            return torch.tensor([0., 0., 0., 1.])
    class ConstantValue:
        def construct_network(self):
            pass
        def get_value(self, state):
            return 0.
    factory = partial(AdasubsPlanner, generators_k_list=[1]) if planner == "adaptive" else UniversalPropositionalPlanner
    solver = Solver(20, cast(Callable[[State], Planner], factory),
                    cast(AdaptiveSubgoalGenerator | SubgoalGenerator, gen),
                    BasicValidator(env, cast(ConditionalLowLevelPolicy, RightPolicy()), 0),
                    cast(Value, ConstantValue()))
    solver.construct_networks()
    experience = solver.solve(from_rows(START))
    assert experience.solution.solved
    assert experience.solution.action_path == [3]
    assert env.is_solved(env.next_state(from_rows(START), 3))
    assert gen.api.metrics["response_requests"] == 1


def test_expansion_budget_applies_even_to_cached_requests(tmp_path, demonstration_file):
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response('{"subgoals": []}')))
    gen = generator(tmp_path, demonstration_file, client=client, max_expansions=1)
    node = SearchTreeNode(from_rows(START), 0, None, None, 1)
    gen.get_subgoals(node)
    with pytest.raises(BudgetExceeded):
        gen.get_subgoals(node)


def test_real_sdk_wire_format_with_luna_prefix_cache(tmp_path, demonstration_file):
    import httpx
    OpenAI = pytest.importorskip("openai").OpenAI
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=dict(id="resp_test", object="response", created_at=0,
            status="completed", model="gpt-6-luna", output=[dict(id="msg_test", type="message", role="assistant",
                status="completed", content=[dict(type="output_text", text=json.dumps(dict(subgoals=[dict(rows=TARGET)])), annotations=[])])],
            usage=dict(input_tokens=10, output_tokens=5, total_tokens=15)))
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = OpenAI(api_key="unit-test-not-a-real-key", http_client=transport, max_retries=0)
        gen = generator(tmp_path, demonstration_file, client=client)
        gen.model = "gpt-6-luna"
        proposals = gen.get_subgoals(SearchTreeNode(from_rows(START), 0, None, None, 1))
    assert len(proposals) == 1
    assert captured[0]["prompt_cache_options"] == dict(mode="explicit", ttl="30m")
    assert captured[0]["input"][-2]["content"][0]["prompt_cache_breakpoint"] == dict(mode="explicit")
    assert captured[0]["store"] is False
