"""Model snapshot verification and matched training configuration."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.policy_relative.prompts import encode_example
from carl.icl.policy_relative.provenance import code_provenance
from carl.icl.policy_relative.types import Arm
from carl.icl.policy_relative.types import Context
from carl.icl.policy_relative.types import FitData


def _model_revision(path: Path, revision: str) -> None:
    # A Hugging Face revision is a 40-character commit, not a SHA-256.
    if (
        not path.is_dir()
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise ValueError(
            "model must be a local directory with an exact 40-character revision"
        )
    metadata = path / "experiment_revision.json"
    if path.name != revision and (
        not metadata.is_file()
        or json.loads(metadata.read_text()).get("revision") != revision
    ):
        raise ValueError("cannot verify the requested local model revision")
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        if not (path / name).is_file():
            raise ValueError(f"local model snapshot is missing {name}")


def _candidate_lengths(
    tokenizer: Any, contexts: tuple[Context, ...], arm: Arm, limit: int
) -> dict[str, int]:
    lengths = sorted(
        len(encode_example(tokenizer, candidate, arm, limit)["input_ids"])
        for context in contexts
        for candidate in context.candidates
    )
    return {
        "count": len(lengths),
        "minimum": lengths[0],
        "median": lengths[len(lengths) // 2],
        "maximum": lengths[-1],
        "limit": limit,
        "truncated": 0,
    }


def _configuration(
    args: argparse.Namespace,
    data: FitData,
    tokenizer: Any,
    device: str,
    demonstration: FitData | None = None,
) -> dict[str, Any]:
    model = Path(args.model).resolve()
    data_path = args.data if Arm(args.arm) != Arm.DEMONSTRATION else args.demo_data
    code_sha256, code_files_sha256 = code_provenance()
    comparable = {
        "data_sha256": file_sha256(data_path),
        "model_revision": args.revision,
        "model_weights_sha256": file_sha256(model / "model.safetensors"),
        "model_config_sha256": file_sha256(model / "config.json"),
        "tokenizer_sha256": file_sha256(model / "tokenizer.json"),
        "controller_hash": data.controller_hash,
        "action_cap": data.action_cap,
        "attempts_per_candidate": data.attempts_per_candidate,
        "seed": args.seed,
        "dtype": args.dtype,
        "threads": getattr(args, "threads", 8),
        "rank": args.rank,
        "lora_targets": ["q_proj", "v_proj"],
        "accumulation": args.accumulation,
        "learning_rate": args.lr,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "max_length": args.max_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        "min_informative_train_contexts": getattr(
            args, "min_informative_train_contexts", 4
        ),
        "min_informative_validation_contexts": getattr(
            args, "min_informative_validation_contexts", 1
        ),
        "train_context_ids": [item.context_id for item in data.train],
        "validation_context_ids": [item.context_id for item in data.validation],
        "train_layout_ids": data.audit["layout_ids_by_split"][DataSplit.TRAIN],
        "validation_layout_ids": data.audit["layout_ids_by_split"][
            DataSplit.VALIDATION
        ],
        "sampling": "equal contexts, candidates sampled by measured weight",
        "loss": "response and EOS tokens only",
        "code_sha256": code_sha256,
        "code_files_sha256": code_files_sha256,
    }
    if demonstration is not None:
        comparable.update(
            {
                "demonstration_sha256": file_sha256(args.demo_data),
                "demonstration_train_context_ids": [
                    item.context_id for item in demonstration.train
                ],
                "demonstration_validation_context_ids": [
                    item.context_id for item in demonstration.validation
                ],
                "train_layout_ids": sorted(
                    set(comparable["train_layout_ids"])
                    | set(demonstration.audit["layout_ids_by_split"][DataSplit.TRAIN])
                ),
                "validation_layout_ids": sorted(
                    set(comparable["validation_layout_ids"])
                    | set(
                        demonstration.audit["layout_ids_by_split"][DataSplit.VALIDATION]
                    )
                ),
                "sampling": "alternating demonstration and controller-weighted draws; D replaces measured with demo",
                "demonstration_fraction": 0.5,
                "validation_sampling": "fixed weighted draws, one per demo context per pool; seeds +2 and +3",
                "validation_draws_per_pool": len(demonstration.validation),
            }
        )
    parity = hashlib.sha256(
        json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    token_lengths = {
        "train": _candidate_lengths(
            tokenizer, data.train, Arm(args.arm), args.max_length
        ),
        "validation": _candidate_lengths(
            tokenizer, data.validation, Arm(args.arm), args.max_length
        ),
    }
    if demonstration is not None:
        token_lengths["demonstration_train"] = _candidate_lengths(
            tokenizer, demonstration.train, Arm(args.arm), args.max_length
        )
        token_lengths["demonstration_validation"] = _candidate_lengths(
            tokenizer, demonstration.validation, Arm(args.arm), args.max_length
        )
    return {
        **comparable,
        "comparability_sha256": parity,
        "arm": args.arm,
        "model": str(model),
        "device": device,
        "token_lengths": token_lengths,
        "enable_thinking": False,
        "command": sys.argv,
        "data_audit": data.audit,
    }


def require_scale_information(
    data: FitData, arm: Arm | str, minimum_train: int = 4, minimum_validation: int = 1
) -> None:
    """Do not present a singleton or all-tied calibration as a scale study."""
    if Arm(arm) in (Arm.UNSCALED, Arm.DEMONSTRATION, Arm.MEASURED):
        return
    if minimum_train < 1 or minimum_validation < 1:
        raise ValueError("minimum informative context counts must be positive")
    for split, minimum in (
        (DataSplit.TRAIN, minimum_train),
        (DataSplit.VALIDATION, minimum_validation),
    ):
        count = data.audit["splits"][split]["contexts_with_distinct_success_durations"]
        if count < minimum:
            raise ValueError(
                f"scaled training needs at least {minimum} {split} contexts with two distinct "
                f"measured successful durations; found {count}"
            )
