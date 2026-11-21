"""Prepare matched LoRA generators from measured policy-relative intents.

This module never trains the controller or calls its API.  A calibration file
contains all candidates in each context, including failures, and one complete
set of measured attempts per candidate.  The supplied ``u`` and ``weight``
are checked against those attempts before successful candidates are used.

Every row requires ``context_id`` and ``policy_measurements`` with
``controller_hash``, ``action_cap``, ``expected_candidates``, ``complete``,
``attempts``, ``success_fraction``, ``mean_success_actions``, ``u`` and
``weight``.  Each attempt has ``status`` and ``primitive_actions_attempted``.
The producer must separately freeze the planned candidate IDs: a declared
candidate count alone cannot prove that the selected pool was not changed.

Generator inputs are restricted to the current observation, task and either
an explicitly requested scale or the missing-scale marker.  Evaluation uses
separate test contexts without reference intents or future observations."""

import argparse
import json
import math
from pathlib import Path

from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.policy_relative.data import load_study_data
from carl.icl.policy_relative.evaluation import evaluate
from carl.icl.policy_relative.resource import resource_check
from carl.icl.policy_relative.training import benchmark
from carl.icl.policy_relative.training import train
from carl.icl.policy_relative.types import Arm
from carl.icl.policy_relative.types import Mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=list(Mode))
    parser.add_argument(
        "--data", help="calibration JSONL or separate test-context JSONL"
    )
    parser.add_argument(
        "--demo-data", help="common checked demonstration pool for D/N/T/U training"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--arm",
        choices=list(Arm),
        help="required except for the unmeasured resource check",
    )
    parser.add_argument(
        "--model", help="verified local model snapshot; required except for inspect"
    )
    parser.add_argument(
        "--revision",
        help="exact local Hugging Face commit; required except for inspect",
    )
    parser.add_argument(
        "--adapter", help="saved generator adapter; required for evaluate"
    )
    parser.add_argument(
        "--requested-scale",
        type=float,
        help="relative u or absolute action duration at evaluation",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora-target", dest="lora_targets", action="append",
                        help="PEFT target module suffix; repeat to override q_proj and v_proj")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pad-to-length", type=int, default=0,
                        help="equalize the model token budget while masking padding")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--min-informative-train-contexts", type=int, default=4)
    parser.add_argument("--min-informative-validation-contexts", type=int, default=1)
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--stop-epoch", type=float)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.threads,
            args.rank,
            args.accumulation,
            args.max_length,
            args.max_new_tokens,
            args.max_steps,
            args.eval_every,
            args.min_informative_train_contexts,
            args.min_informative_validation_contexts,
        )
        < 1
    ):
        parser.error("steps, lengths, threads and accumulation must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be finite and positive")
    if not 0 <= args.pad_to_length <= args.max_length:
        parser.error("pad-to-length must be between zero and max-length")
    if args.require_cuda and args.device == "cpu":
        parser.error("--require-cuda conflicts with --device cpu")
    if args.mode == Mode.RESOURCE_CHECK and args.arm is not None:
        parser.error(
            "resource-check uses diagnostic scale 0.5 and does not accept --arm"
        )
    if args.mode != Mode.RESOURCE_CHECK and args.arm is None:
        parser.error("this mode requires --arm")
    if not args.data and not (
        args.arm == Arm.DEMONSTRATION
        and args.mode in (Mode.INSPECT, Mode.BENCHMARK, Mode.TRAIN)
    ):
        parser.error(
            "--data is required except for D inspection, benchmarking and training"
        )
    if (
        args.mode in (Mode.INSPECT, Mode.BENCHMARK, Mode.TRAIN)
        and args.arm in (Arm.DEMONSTRATION, Arm.MEASURED, Arm.ABSOLUTE, Arm.RELATIVE)
        and not args.demo_data
    ):
        parser.error("D/N/T/U require --demo-data")
    if (
        args.mode in (Mode.BENCHMARK, Mode.TRAIN)
        and args.arm in (Arm.DEMONSTRATION, Arm.MEASURED, Arm.ABSOLUTE, Arm.RELATIVE)
        and args.accumulation % 2
    ):
        parser.error(
            "D/N/T/U require even gradient accumulation for an exact 50/50 mix"
        )
    if args.mode != Mode.INSPECT and not all((args.model, args.revision)):
        parser.error(
            "resource-check, benchmark, train and evaluate require --model and --revision"
        )
    if args.arm == Arm.BASE and args.mode != Mode.EVALUATE:
        parser.error("the base arm is available for evaluation only")
    if args.mode == Mode.EVALUATE and args.arm == Arm.BASE and args.adapter is not None:
        parser.error("the base arm must not load an adapter")
    if args.mode == Mode.EVALUATE and args.arm != Arm.BASE and not args.adapter:
        parser.error("evaluate requires --adapter")
    if args.mode == Mode.INSPECT:
        data, demonstration = load_study_data(args)
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("use a fresh inspection output")
        atomic_json(
            output,
            {
                "data_sha256": file_sha256(
                    args.demo_data if args.arm == Arm.DEMONSTRATION else args.data
                ),
                "demonstration_sha256": file_sha256(args.demo_data)
                if demonstration
                else None,
                "arm": args.arm,
                "controller_hash": data.controller_hash,
                "action_cap": data.action_cap,
                "attempts_per_candidate": data.attempts_per_candidate,
                "audit": data.audit,
            },
        )
        print(json.dumps(data.audit, indent=2))
    elif args.mode == Mode.RESOURCE_CHECK:
        resource_check(args)
    elif args.mode == Mode.BENCHMARK:
        benchmark(args)
    elif args.mode == Mode.TRAIN:
        train(args)
    else:
        evaluate(args)
