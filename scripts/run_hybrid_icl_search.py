"""Prepare or run CaRL search with a prompted subgoal generator.

Default: inspect initial requests without API calls or checkpoint loading.
Run as `uv run --extra icl python -m scripts.run_hybrid_icl_search --help`.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from transformers import BertForSequenceClassification

from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows, level_key, to_rows, validate_subgoal
from carl.icl.config import ExperimentConfig, Limits, RoleSettings
from carl.icl.data import prepare_data
from carl.icl.experiment import create_inference_api, prepare_credentials
from carl.icl.generator import OpenAISubgoalGenerator
from carl.inference_components.conditional_low_level_policy import TransformerConditionalLowLevelPolicy
from carl.inference_components.subgoal_generator import AdaptiveSubgoalGenerator
from carl.inference_components.validator import BasicValidator
from carl.inference_components.value import TransformerValue
from carl.planners.adasubs import AdasubsPlanner, UniversalPropositionalPlanner
from carl.planners.base import Planner
from carl.solver.nodes import SearchTreeNode
from carl.solver.subgoal_search import Solver
from carl.utils.aliases import State

ROOT = Path(__file__).resolve().parents[1]


def read_config(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.config)
    config = json.loads((path if path.is_absolute() else ROOT / path).read_text())
    for name in (
        "model",
        "planner",
        "selection",
        "max_requests",
        "max_nodes",
        "instances",
        "num_subgoals",
        "examples_per_request",
        "output",
    ):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
    for name in (
        "trajectories_path",
        "demonstrations_path",
        "instances_path",
        "easy_instances_path",
        "cllp_path",
        "value_path",
        "cache_dir",
        "output",
    ):
        path = Path(config[name]).expanduser()
        config[name] = str(path if path.is_absolute() else ROOT / path)
    if args.reasoning_effort is not None:
        config["reasoning_effort"] = None if args.reasoning_effort == "omit" else args.reasoning_effort
    if "roles" in config:
        roles = RoleSettings.model_validate(config["roles"])
        if roles.policy is not None or roles.value is not None or roles.generator is None:
            raise ValueError("Hybrid search configures roles.generator and loads policy and value from local checkpoints")
        if getattr(args, "model", None) is not None:
            raise ValueError("Set roles.generator.model and its revision together in the hybrid config")
        generator = roles.generator.model_dump()
        if args.reasoning_effort is not None:
            generator["reasoning_effort"] = config["reasoning_effort"] or "none"
        roles = RoleSettings.model_validate({"generator": generator})
        assert roles.generator is not None
        if config["selection"] == "embeddings" and not roles.generator.capabilities.embeddings:
            raise ValueError("Generator endpoint does not support embeddings")
        config["model"] = roles.generator.model
        config["roles"] = roles.model_dump(mode="json")
    if config["instances"] < 1 or config["max_nodes"] < 1:
        raise ValueError("Instances and node budgets must be positive")
    return config


def run(args: argparse.Namespace) -> int:
    config = read_config(args)
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    tokenizer = SokobanTokenizer(
        size_of_board=tuple(config["board_size"]), cut_distance=150, type_of_value_training="regression"
    )
    env = SokobanEnv(tokenizer, num_boxes=config["num_boxes"])
    if args.prepare_data:
        provenance = prepare_data(config, env)
        print(
            f"Prepared disjoint demonstrations/evaluation; replayed {provenance['replayed_transitions']} transitions",
            flush=True,
        )
    if args.easy:
        config["instances_path"] = config["easy_instances_path"]
    try:
        dataset = json.loads(Path(config["instances_path"]).read_text())
        demos = json.loads(Path(config["demonstrations_path"]).read_text())["demonstrations"]
    except FileNotFoundError:
        raise ValueError(
            "Prepare inputs first with --prepare-data, or supply demonstrations_path/instances_path in the config"
        ) from None
    instances = dataset["instances"][: config["instances"]]
    if len(instances) < config["instances"]:
        raise ValueError("Not enough prepared instances; rerun --prepare-data")
    demo_levels = {level_key(from_rows(example["state"])) for example in demos}
    for instance in instances:
        state = from_rows(instance["state"])
        validate_subgoal(state, state)
        if state.shape[:2] != tuple(config["board_size"]):
            raise ValueError("Instance shape differs from configured board_size")
        if int(np.isin(state.argmax(-1), [3, 4]).sum()) != config["num_boxes"]:
            raise ValueError("Instance box count differs from configured num_boxes")
        if level_key(state) in demo_levels:
            raise ValueError("Evaluation layout overlaps the demonstrations")
    provider = "openai" if args.execute else "mock" if args.mock else "replay"
    if args.mock and config["selection"] == "embeddings":
        raise ValueError("--mock supports --selection direct; embeddings require --execute or cached --replay")
    preview_only = not (args.execute or args.mock or args.replay)
    inference = ExperimentConfig(
        model=config["model"],
        roles=config.get("roles"),
        cache_dir=config["cache_dir"],
        limits=Limits(
            max_requests=config["max_requests"],
            timeout_seconds=config["timeout_seconds"],
            wall_seconds=config["max_wall_seconds"],
            max_output_tokens=config["max_requests"] * config["max_output_tokens"],
        ),
    )
    transport = inference.explicit_role("generator")
    if args.execute and transport is not None:
        prepare_credentials(inference, provider)
    api = create_inference_api(
        inference, provider, transport=transport, preview_only=preview_only,
    )
    generator = OpenAISubgoalGenerator(
        env=env,
        model=config["model"],
        demonstrations_path=config["demonstrations_path"],
        api=api,
        planner=config["planner"],
        generator_k_list=config["generator_k_list"],
        num_subgoals=config["num_subgoals"],
        examples_per_request=config["examples_per_request"],
        selection=config["selection"],
        embedding_model=config["embedding_model"],
        max_output_tokens=config["max_output_tokens"],
        reasoning_effort=(
            config["reasoning_effort"] if transport is None or args.reasoning_effort == "omit"
            else transport.reasoning_effort
        ),
        transport=transport,
        controller_cap=config["controller_cap"],
        max_expansions=config["max_expansions"],
    )
    output = Path(config["output"]) / f"{time.time_ns()}_{provider}_{config['planner']}_{config['selection']}"
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    if preview_only:
        payloads = []
        for instance in instances:
            for k in generator.generator_k_list or [None]:
                node = SearchTreeNode(from_rows(instance["state"]), 0, None, None, next_expand_with_k_generator=k)
                payloads.append(
                    {"instance_id": instance["id"], "payload": generator.build_payload(node, resolve_embeddings=False)}
                )
        write_json(
            output / "prepared_requests.json",
            {
                "mode": "offline_preview",
                "embedding_selection_resolved": False,
                "requests": payloads,
                "note": "Initial-node requests only. Search requests depend on validated model proposals. Embeddings preview uses file order.",
            },
        )
        print(f"Offline preview: {output / 'prepared_requests.json'}\nNo API requests or checkpoint loads.")
        return 0
    # Check both paths before CaRL's retrying model loader gets involved.
    for name in ("cllp_path", "value_path"):
        if not (Path(config[name]) / "config.json").is_file():
            raise ValueError(f"Missing local checkpoint: {config[name]}")
    device = torch.device(config["device"])
    cllp = TransformerConditionalLowLevelPolicy(
        partial(BertForSequenceClassification.from_pretrained, local_files_only=True), config["cllp_path"], env
    )
    value = TransformerValue(
        partial(BertForSequenceClassification.from_pretrained, local_files_only=True),
        config["value_path"],
        env,
        type_of_evaluation="regression",
        noise_variance=0.0,
    )
    cllp.device = device
    value.device = device
    # BasicValidator's fallback adds one; subtract it here for an exact universal cap.
    validator = BasicValidator(env, cllp, budget_for_achieving_subgoal=config["controller_cap"] - 1)

    def planner(state: State) -> Planner:
        if not isinstance(state, np.ndarray):
            raise ValueError("Only Sokoban array states are supported")
        if config["planner"] == "adaptive":
            return AdasubsPlanner(state, generators_k_list=generator.generator_k_list)
        return UniversalPropositionalPlanner(state)

    solver = Solver(config["max_nodes"], planner, cast(AdaptiveSubgoalGenerator, generator), validator, value)
    solver.construct_networks()
    # Exclude local checkpoint loading from the API/search wall-time budget.
    api.started = time.perf_counter()
    results: list[dict[str, Any]] = []
    status = "completed"
    exit_code = 0
    for instance in instances:
        started = time.perf_counter()
        state = from_rows(instance["state"])
        result: dict[str, Any] = {"id": instance["id"], "provider": provider}
        try:
            if env.is_solved(state):
                result.update(solved=True, finish_reason="already_solved", action_path=[])
            else:
                experience = solver.solve(state)
                solution, info = experience.solution, experience.search_info
                result.update(
                    solved=solution.solved,
                    finish_reason=info.finished_reason,
                    action_path=solution.action_path,
                    subgoal_path=[to_rows(s) for s in solution.subgoal_path] if solution.subgoal_path else [],
                    low_level_nodes=info.low_level_nodes_visited,
                    accepted=info.high_level_nodes_valid,
                    unreachable=info.high_level_nodes_unreachable,
                    duplicate_proposals=info.proposal_duplicates,
                    reachable_per_k=info.subgoals_reachable_rate_per_k,
                )
                # Validate every reported solution independently by replaying primitive actions.
                if solution.solved:
                    replay = state.copy()
                    for action in solution.action_path or []:
                        replay = env.next_state(replay, action)
                    if not env.is_solved(replay):
                        raise RuntimeError("Solver action path failed independent replay")
                    result["solution_replayed"] = True
        except (BudgetExceeded, RequestFailed) as exc:
            status = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "request_failed"
            result.update(solved=False, finish_reason=status, error=str(exc))
            exit_code = 0 if isinstance(exc, BudgetExceeded) else 1
        result["seconds"] = time.perf_counter() - started
        results.append(result)
        summary = {
            "status": status,
            "provider": provider,
            "model": config["model"],
            "planner": config["planner"],
            "selection": config["selection"],
            "requested_instances": len(instances),
            "attempted_instances": len(results),
            "solved": sum(r["solved"] for r in results),
            "api": api.metrics,
            "generator": generator.metrics,
            "results": results,
            "caveat": "Mock uses legal one-step successors, not an LLM. This small within-shard split does not establish generalisation.",
        }
        write_json(output / "summary.json", summary)
        print(json.dumps(result), flush=True)
        if status != "completed":
            break
    print(f"Results: {output / 'summary.json'}", flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icl/sokoban_search.json")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Export demonstrations/instances from local trajectories, with replay checks",
    )
    parser.add_argument(
        "--easy",
        action="store_true",
        help="Use disjoint evaluation boards one move from solved, for an interface diagnostic",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Use the configured generator API")
    mode.add_argument(
        "--mock", action="store_true", help="Full solver with explicitly labelled legal-successor mock proposals"
    )
    mode.add_argument(
        "--replay", action="store_true", help="Full solver using completed live response/embedding caches only"
    )
    parser.add_argument("--model")
    parser.add_argument("--planner", choices=["adaptive", "universal"])
    parser.add_argument("--selection", choices=["direct", "embeddings"])
    parser.add_argument("--reasoning-effort", choices=["omit", "none", "minimal", "low", "medium", "high"])
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--num-subgoals", type=int)
    parser.add_argument("--examples-per-request", type=int)
    parser.add_argument("--instances", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, RequestFailed) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
