#!/usr/bin/env python3
"""Compare local CPU batch decisions to CaRL's public policy components.

Only pre-split development boards are accepted. No training, API or GPU is used.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from transformers import BertForSequenceClassification

from carl.environment.training_goal import TrainingGoal
from carl.icl.boards import state_hash
from carl.icl.data_sources.sokoban import environment
from carl.inference_components.conditional_low_level_policy import TransformerConditionalLowLevelPolicy
from carl.inference_components.policy import TransformerPolicy
from scripts.run_overnight_cpu_policy import CHECKPOINTS
from scripts.run_overnight_cpu_policy import ModelId
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import materialise_pairs
from scripts.run_overnight_cpu_policy import model_predictor
from scripts.run_overnight_cpu_policy import policy_predictor
from scripts.run_overnight_cpu_policy import read_jsonl
from scripts.run_overnight_cpu_policy import select_groups
from scripts.training.calibration import pair_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--boards-per-family", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if min(args.boards_per_family, args.threads, args.batch_size) < 1:
        parser.error("Board, thread and batch counts must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    groups = select_groups(read_jsonl(args.pairs), "development", args.boards_per_family, max_contexts=1)
    pairs = []
    for group in groups:
        eligible = materialise_pairs(group)
        # Vary offsets across boards, but never pull a held-out pair.
        desired = (1, 4, 16)[group["board_rank"] % 3]
        pair = next((item for item in eligible if item.k == desired), eligible[0])
        pairs.append((group["family"], pair))
    if {family for family, _ in pairs} != {"12x12x4", "12x12x10"}:
        raise ValueError("Both development families are required")
    model_ids = (ModelId.CLLP_8, ModelId.CLLP_32, ModelId.POLICY)
    factory = partial(BertForSequenceClassification.from_pretrained, local_files_only=True, use_safetensors=False)
    records = []
    for model_id in model_ids:
        first_env = environment(pairs[0][1].source)
        if model_id == ModelId.POLICY:
            component = TransformerPolicy(factory, first_env, str(CHECKPOINTS[model_id]), n_actions=1)
        else:
            component = TransformerConditionalLowLevelPolicy(factory, str(CHECKPOINTS[model_id]), first_env)
        component.construct_network()
        model = component.get_network().to("cpu").float().eval()
        model.requires_grad_(False)
        states = [pair.source for _, pair in pairs]
        targets = [pair.target for _, pair in pairs]
        tokenizer = first_env.tokenizer
        if model_id == ModelId.POLICY:
            batch_actions = policy_predictor(model, tokenizer, args.batch_size)(states, targets)
        else:
            batch_actions = model_predictor(model, tokenizer, torch.device("cpu"), args.batch_size)(states, targets)
        chunks = []
        for start in range(0, len(states), args.batch_size):
            batch_states = states[start:start + args.batch_size]
            batch_targets = targets[start:start + args.batch_size]
            if model_id == ModelId.POLICY:
                tokens = torch.cat([tokenizer.x_y_tokenizer(state, 0, TrainingGoal.POLICY)[0]
                                    for state in batch_states])
            else:
                tokens = pair_tokens(tokenizer, batch_states, batch_targets)
            with torch.inference_mode():
                chunks.append(model(tokens).logits.detach().cpu())
        all_batch_logits = torch.cat(chunks)
        for index, (family, pair) in enumerate(pairs):
            env = environment(pair.source)
            component.env = env
            if model_id == ModelId.POLICY:
                public = component.get_actions(pair.source)[0]
                public_logits = public.generation_metadata["action_probs"].detach().cpu()
                public_action = int(public.action)
            else:
                public_logits = component.get_action(pair.source, pair.target).detach().cpu()
                public_action = int(env.distribution_to_action(public_logits))
            batch_action, batch_logits = batch_actions[index], all_batch_logits[index]
            equal = bool(torch.allclose(public_logits, batch_logits, rtol=1e-5, atol=1e-6))
            maximum_difference = float(torch.max(torch.abs(public_logits - batch_logits)))
            records.append({"model_id": model_id, "family": family, "layout_id": pair.layout_id,
                            "pair_id": pair.pair_id, "source_hash": state_hash(pair.source),
                            "target_hash": state_hash(pair.target), "k": pair.k,
                            "public_action": public_action, "batch_action": batch_action,
                            "logits_allclose": equal, "max_abs_logit_difference": maximum_difference,
                            "action_match": public_action == batch_action})
        del component, model
    summary = {"schema_version": 1, "split": "development", "device": "cpu", "models": model_ids,
               "effective_batch_size": min(args.batch_size, len(pairs)),
               "families": sorted({family for family, _ in pairs}), "cases": len(records),
               "all_logits_close": all(item["logits_allclose"] for item in records),
               "all_actions_match": all(item["action_match"] for item in records), "records": records}
    atomic_json(args.output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}), flush=True)
    if not summary["all_logits_close"] or not summary["all_actions_match"]:
        raise SystemExit("CPU batched policy differs from the public CaRL component")


if __name__ == "__main__":
    main()
