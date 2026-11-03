"""Bounded, local feasibility tests for policy-relative subgoal conditioning.

The CNN is an explicit-state ablation, not the proposal's latent planner. The
MLP tests its weighted latent MSE using cached features from the frozen CLLP;
those features do not make the existing board-input controller latent-compatible.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import torch
from torch import nn
from transformers import BertForSequenceClassification

from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer


from scripts.training.calibration import (
    weighted_midpoints,
    level_key,
    valid_board,
    pair_tokens,
    rollout_trials,
    encode_pairs,
    prepare,
)
from scripts.training.small_models import SmallBoardGenerator, SmallLatentGenerator, fit

__all__ = [
    "weighted_midpoints",
    "level_key",
    "valid_board",
    "pair_tokens",
    "rollout_trials",
    "encode_pairs",
    "prepare",
    "SmallBoardGenerator",
    "SmallLatentGenerator",
    "fit",
    "write_csv",
    "main",
]


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_1.pkl"),
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("rl-data/validation/sokoban/components/full_data/cllp/8/checkpoint-167585")
    )
    for name, default in [
        ("boards", 48),
        ("contexts", 2),
        ("window", 16),
        ("candidates", 8),
        ("repeats", 4),
        ("cap", 20),
        ("policy-batch", 32),
        ("batch", 32),
        ("steps", 400),
        ("eval-every", 100),
        ("seed", 20261029),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("Choose a fresh output directory; completed experiments are immutable")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    tokenizer = SokobanTokenizer()
    env = SokobanEnv(tokenizer)
    policy = BertForSequenceClassification.from_pretrained(args.policy, local_files_only=True).eval()
    cast(nn.Module, policy).cuda()
    policy.requires_grad_(False)
    dataset, calibration = prepare(args, policy, tokenizer, env)
    joblib.dump(dataset, args.output / "calibrated_dataset.joblib")
    (args.output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(
        "CALIBRATION",
        json.dumps({k: v for k, v in calibration.items() if k not in ["contexts", "records"]}),
        flush=True,
    )
    # Keep the controller out of GPU memory while optimising the small generators.
    policy_parameters = sum(p.numel() for p in policy.parameters())
    del policy
    torch.cuda.empty_cache()
    summaries, history = [], []
    for architecture in ["cnn", "latent_mlp"]:
        for condition in ["relative", "absolute", "none"]:
            summary, logs = fit(args, dataset, architecture, condition)
            summaries.append(summary)
            history.extend(logs)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    provenance = dict(
        config=config,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        policy_parameters=policy_parameters,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.policy.iterdir()
            if p.name in ["config.json", "model.safetensors", "pytorch_model.bin"]
        },
    )
    report = dict(
        provenance=provenance,
        calibration={k: v for k, v in calibration.items() if k not in ["contexts", "records"]},
        runs=summaries,
        total_seconds=time.perf_counter() - started,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.output / "learning_curves.csv", history)
    write_csv(
        args.output / "metrics.csv",
        [
            dict(
                parameters=r["parameters"],
                seconds=r["seconds"],
                peak_allocated_mib=r["peak_allocated_mib"],
                initial_train_loss=r["initial"]["train_loss"],
                final_train_loss=r["final"]["train_loss"],
                initial_val_loss=r["initial"]["val_loss"],
                final_val_loss=r["final"]["val_loss"],
                **r["final"]
            )
            for r in summaries
        ],
    )
    print("COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
