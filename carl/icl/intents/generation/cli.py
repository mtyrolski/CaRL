"""Command-line interface for local intent generation."""

import argparse
from collections.abc import Callable

from carl.icl.intents.generation.data import DEFAULT_MODEL
from carl.icl.intents.generation.data import DEFAULT_REVISION
from carl.icl.intents.generation.data import RunMode
from carl.icl.intents.generation.data import TrainingAugmentation


def run_cli(
    benchmark: Callable[[argparse.Namespace], None],
    train: Callable[[argparse.Namespace], None],
    evaluate: Callable[[argparse.Namespace], None],
    *,
    description: str | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("mode", choices=tuple(RunMode))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision", help="exact model commit; required when using another model"
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--augmentation",
        choices=tuple(TrainingAugmentation),
        default=TrainingAugmentation.NONE,
    )
    parser.add_argument(
        "--pad-to-length",
        type=int,
        default=0,
        help="pad complete training examples to this common token budget; never truncate",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--validation-boards", type=int, default=12)
    parser.add_argument("--validation-contexts-per-board", type=int, default=4)
    parser.add_argument("--test-boards", type=int, default=0)
    parser.add_argument(
        "--test-ids", help="JSON file with the pre-frozen record_ids array"
    )
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--adapter")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted evaluation after validating its configuration",
    )
    args = parser.parse_args()
    if args.revision is None:
        if args.model != DEFAULT_MODEL:
            parser.error(
                "--revision is required for a non-default model or local model directory"
            )
        args.revision = DEFAULT_REVISION
    if (
        args.accumulation < 1
        or args.max_length < 1
        or args.eval_every < 1
        or args.validation_contexts_per_board < 1
    ):
        parser.error(
            "accumulation, max-length, eval-every and validation-contexts-per-board must be positive"
        )
    if args.pad_to_length < 0 or args.pad_to_length > args.max_length:
        parser.error("pad-to-length must be between zero and max-length")
    {RunMode.BENCHMARK: benchmark, RunMode.TRAIN: train, RunMode.EVALUATE: evaluate}[
        args.mode
    ](args)
