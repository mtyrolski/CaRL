"""Prepare, run/cache, and locally evaluate matched Sokoban ICL experiments.

Default execution is offline. Use --execute for OpenAI Responses calls, or
--mock for an explicitly labelled copy-state plumbing check. No API keys are
written or included in cached payloads. Production data and answers stay separate.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import joblib
import numpy as np

from scripts.training.policy_relative_smoke import level_key, valid_board, write_csv

SYMBOLS = "#-.*$@+"
SCHEMA = {
    "type": "object",
    "properties": {"rows": {"type": "array", "minItems": 12, "maxItems": 12, "items": {"type": "string", "pattern": "^[#.$*@+-]{12}$"}}},
    "required": ["rows"],
    "additionalProperties": False,
}
BASE_PROMPT = """Propose one intermediate Sokoban state towards the supplied task goal, following the demonstrations.
Each board has 12 rows of 12 characters: # wall, - floor, . goal, $ box, * box on goal,
@ player, + player on goal. Preserve all walls and goal squares, exactly four boxes
and exactly one player. Boxes are pushed, never pulled. Propose a reachable state
different from the start. Return only the requested JSON board, not an action plan.
The controller has a separate fixed execution cap L={cap} actions."""
CONDITION_PROMPTS = {
    "relative": (
        "The requested scale u is the success-weighted percentile of mean successful first-hit execution time within this starting context. "
        "Learn its contextual meaning from examples. u is not a success probability or a trajectory offset."
    ),
    "absolute": "The requested scale d is the mean successful first-hit execution time in actions. Learn its contextual meaning from examples. d is not necessarily a trajectory offset.",
    "none": "No temporal-scale input is provided; propose an intermediate target as in the demonstrations.",
}


def board_rows(tiles):
    return ["".join(SYMBOLS[int(v)] for v in row) for row in tiles]


def parse_board(payload):
    if not isinstance(payload, dict) or set(payload) != {"rows"}:
        raise ValueError("Expected only the rows field")
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != 12 or any(not isinstance(r, str) or len(r) != 12 for r in rows):
        raise ValueError("Expected 12 rows of 12 characters")
    if any(c not in SYMBOLS for row in rows for c in row):
        raise ValueError("Unknown tile symbol")
    return np.array([[SYMBOLS.index(c) for c in row] for row in rows], dtype=np.int64)


def public_query(example, condition):
    public = {"state": example["state"], "goal": example["goal"]}
    if condition == "relative":
        public["u"] = example["u"]
    elif condition == "absolute":
        public["d"] = example["d"]
    elif condition != "none":
        raise ValueError("Unknown condition")
    return public


def build_messages(case, demos, condition, cap):
    messages = [{"role": "system", "content": BASE_PROMPT.format(cap=cap) + "\n" + CONDITION_PROMPTS[condition]}]
    for example in demos:
        messages.extend([{"role": "user", "content": json.dumps(public_query(example, condition))}, {"role": "assistant", "content": json.dumps({"rows": example["target"]})}])
    messages.append({"role": "user", "content": json.dumps(public_query(case, condition))})
    return messages


def prepare(config):
    dataset = joblib.load(Path(config["dataset"]) / "calibrated_dataset.joblib")
    rows = dataset["rows"]
    rng = np.random.default_rng(config["seed"])

    def example(i):
        r = rows[i]
        return dict(
            row_index=i,
            context=r["context"],
            trajectory=r["trajectory"],
            split=r["split"],
            state=board_rows(dataset["starts"][i]),
            goal=board_rows(dataset["goals"][i]),
            target=board_rows(dataset["targets"][i]),
            u=r["u"],
            d=r["mean_hit_time"],
        )

    train_boards = sorted({r["trajectory"] for r in rows if r["split"] == "train"})
    chosen = rng.choice(train_boards, config["few_shot_examples"], replace=False)
    demos = [example(int(rng.choice([i for i, r in enumerate(rows) if r["trajectory"] == board]))) for board in chosen]
    contexts = sorted({r["context"] for r in rows if r["split"] == "test"})
    query_contexts = rng.choice(contexts, config["query_contexts"], replace=False)
    cases = []
    for j, c in enumerate(query_contexts):
        available = [i for i, r in enumerate(rows) if r["context"] == c]
        # Match each condition to the same reference target; do not send that
        # target, its success measurements, or the other coordinate in the query.
        request = 0.5 if j % 2 == 0 else 0.9
        cases.append(example(min(available, key=lambda i: abs(rows[i]["u"] - request))))
    train_levels = {level_key(np.eye(7)[parse_board({"rows": d["state"]})]) for d in demos}
    assert all(level_key(np.eye(7)[parse_board({"rows": c["state"]})]) not in train_levels for c in cases)
    return {"demonstrations": demos, "queries": cases}


def make_requests(config, prepared):
    requests = []
    for case in prepared["queries"]:
        for condition in config["conditions"]:
            payload = dict(
                model=config["model"],
                input=build_messages(case, prepared["demonstrations"], condition, config["execution_cap"]),
                text={"format": {"type": "json_schema", "name": "sokoban_subgoal", "strict": True, "schema": SCHEMA}},
                max_output_tokens=config["max_output_tokens"],
                store=False,
            )
            if config.get("reasoning_effort") is not None:
                payload["reasoning"] = {"effort": config["reasoning_effort"]}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            requests.append(dict(id=digest, condition=condition, context=case["context"], payload=payload))
    if len(requests) > config["max_requests"]:
        raise ValueError("Request count exceeds configured budget")
    # Deliberately conservative ASCII byte-based estimate; actual token usage is
    # saved per response. This is not a provider billing guarantee.
    upper = sum((len(json.dumps(r["payload"]).encode()) + 256) * config["input_usd_per_million"] / 1e6 + config["max_output_tokens"] * config["output_usd_per_million"] / 1e6 for r in requests)
    if upper > config["max_estimated_cost_usd"]:
        raise ValueError("Estimated cost exceeds configured budget")
    return requests, upper


def execute(config, prepared, requests, output, *, live=False, mock=False):
    if live and mock:
        raise ValueError("Choose live or mock")
    provider = "openai" if live else "mock_copy"
    cache = output / provider
    cache.mkdir(parents=True, exist_ok=True)
    api = None
    if live:
        from carl.icl.api import CachedOpenAI

        api = CachedOpenAI(output / "guarded_api_cache", provider="openai", max_requests=config["max_requests"], timeout_seconds=config["timeout_seconds"])
    results = []
    for req in requests:
        path = cache / (req["id"] + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["status"] != "error":
                if live and api is not None and api.ledger is not None:
                    api.ledger.event(api.stage, "logical_request", "responses", req["id"], api.budget_tag)
                    api.ledger.event(api.stage, "live_cache_hit", "responses", req["id"], api.budget_tag)
                results.append(cached)
                continue
        case = next(c for c in prepared["queries"] if c["context"] == req["context"])
        start = time.perf_counter()
        record = {k: req[k] for k in ["id", "condition", "context"]}
        record["provider"] = provider
        if mock:
            record.update(status="completed", output_text=json.dumps({"rows": case["state"]}), usage={"input_tokens": 0, "output_tokens": 0}, model="mock_copy")
        else:
            if api is None:
                raise ValueError("A live client is required unless mock mode is enabled")
            try:
                response = api.call("responses", req["payload"])
                record.update(status=response["status"], output_text=response["output_text"], usage=response.get("usage", {}), response_id=response.get("response_id"), model=response.get("model"))
            except Exception as exc:
                # Never persist raw exception text, which may echo credentials.
                record.update(status="error", error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
                record["seconds"] = time.perf_counter() - start
                path.write_text(json.dumps(record, indent=2) + "\n")
                results.append(record)
                if record["http_status"] in (401, 403):
                    break
                continue
        record["seconds"] = time.perf_counter() - start
        path.write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        print(provider, req["condition"], req["context"], record["status"], flush=True)
    return results


def evaluate(config, prepared, results, output):
    metrics = []
    starts = []
    targets = []
    executable = []
    for record in results:
        case = next(c for c in prepared["queries"] if c["context"] == record["context"])
        source = parse_board({"rows": case["state"]})
        row = {k: record[k] for k in ["id", "condition", "context", "provider", "status"]}
        row.update(format_valid=False, board_valid=False, self_copy=False, exact_reference_target=False, execution_trials=0, execution_hits=0, attempted_actions=0)
        if record["status"] == "completed":
            try:
                pred = parse_board(json.loads(record["output_text"]))
                row["format_valid"] = True
                row["board_valid"] = valid_board(pred, source)
                row["self_copy"] = bool(np.array_equal(pred, source))
                row["exact_reference_target"] = bool(np.array_equal(pred, parse_board({"rows": case["target"]})))
                if row["board_valid"] and not row["self_copy"]:
                    executable.append(len(metrics))
                    starts.append(np.eye(7, dtype=np.float32)[source])
                    targets.append(np.eye(7, dtype=np.float32)[pred])
            except (ValueError, TypeError, KeyError):
                pass
        usage = record.get("usage", {})
        row.update(input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0))
        row["estimated_cost_usd"] = (row["input_tokens"] * config["input_usd_per_million"] + row["output_tokens"] * config["output_usd_per_million"]) / 1e6
        metrics.append(row)
    if executable:
        import torch
        from transformers import BertForSequenceClassification
        from scripts.training.policy_relative_smoke import rollout_trials
        from carl.environment.sokoban.env import SokobanEnv
        from carl.environment.sokoban.tokenizer import SokobanTokenizer

        torch.set_num_threads(4)
        source = json.loads((Path(config["dataset"]) / "summary.json").read_text())
        tok = SokobanTokenizer()
        env = SokobanEnv(tok)
        policy = BertForSequenceClassification.from_pretrained(source["provenance"]["config"]["policy"], local_files_only=True).eval()
        cast(torch.nn.Module, policy).cuda()
        for index, start, target in zip(executable, starts, targets):
            hit, info = rollout_trials(policy, tok, env, [start], [target], repeats=config["execution_trials"], cap=config["execution_cap"], seed=config["seed"], batch_size=32)
            metrics[index].update(execution_trials=int(hit.size), execution_hits=int((hit > 0).sum()), attempted_actions=info["attempted_actions"])
    write_csv(output / "metrics.csv", metrics)
    groups = []
    for condition in config["conditions"]:
        r = [x for x in metrics if x["condition"] == condition]
        groups.append(
            dict(
                condition=condition,
                responses=len(r),
                format_valid=sum(x["format_valid"] for x in r),
                board_valid=sum(x["board_valid"] for x in r),
                self_copies=sum(x["self_copy"] for x in r),
                execution_trials=sum(x["execution_trials"] for x in r),
                execution_hits=sum(x["execution_hits"] for x in r),
            )
        )
    summary = dict(
        config=config,
        provider=metrics[0]["provider"] if metrics else None,
        groups=groups,
        estimated_cost_usd=sum(r["estimated_cost_usd"] for r in metrics),
        caveat="Tiny plumbing test; no claim of scale calibration, planning success, or transfer.",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = [
        "# Sokoban ICL smoke results",
        "",
        f"Provider: {summary['provider']}. Estimated token cost: ${summary['estimated_cost_usd']:.5f}.",
        "",
        "| Condition | Responses | Valid boards | Self-copies | Execution hits / trials |",
        "|---|---:|---:|---:|---:|",
    ]
    text += [f"| {r['condition']} | {r['responses']} | {r['board_valid']} | {r['self_copies']} | {r['execution_hits']} / {r['execution_trials']} |" for r in groups]
    text += ["", summary["caveat"], "Self-copies are not counted as useful executed subgoals. Zero trials means no valid nontrivial board was available."]
    (output / "summary.md").write_text("\n".join(text) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/icl/sokoban_openai_smoke.json"))
    p.add_argument("--output", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--mock", action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.output:
        config["output"] = str(args.output)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare(config)
    requests, estimate = make_requests(config, prepared)
    (output / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    (output / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Prepared {len(requests)} requests; conservative estimated ceiling ${estimate:.4f}", flush=True)
    if not (args.execute or args.mock):
        print("Offline preparation complete. Use --mock to test plumbing or --execute for Responses API calls.")
        return
    results = execute(config, prepared, requests, output, live=args.execute, mock=args.mock)
    print(json.dumps(evaluate(config, prepared, results, output)), flush=True)


if __name__ == "__main__":
    main()
