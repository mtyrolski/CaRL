"""Evaluate saved ICL trials and build diagnostics offline."""
import argparse
from pathlib import Path
from typing import Final

from carl.icl.evaluation.data import EvaluationConfig, load_evaluation_config
from carl.icl.evaluation.matched import evaluate_matched
from carl.icl.evaluation.report import evaluate

DEFAULT_CONFIG: Final[str] = "configs/icl/evaluation/mock.json"


def main() -> int:
    """Load evaluation settings with optional input/output overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--config", help=f"Legacy evaluation config (default {DEFAULT_CONFIG})")
    inputs.add_argument("--matched", type=Path, help="Validated matched sweep JSON")
    parser.add_argument(
        "--runs", nargs="+", help="Condition or matrix directories; overrides run_dirs"
    )
    parser.add_argument("--output", help="Override the output directory")
    parser.add_argument(
        "--formats",
        nargs="*",
        choices=("png", "svg", "pdf"),
        help="Optional static exports via Kaleido",
    )
    parser.add_argument("--bootstrap-seed", type=int, help="Resampling seed for --matched")
    parser.add_argument("--bootstrap-replicates", type=int, help="Resampling count for --matched")
    args = parser.parse_args()
    if args.matched is not None:
        if not args.output or args.runs is not None or args.formats is not None:
            parser.error("--matched requires --output and cannot be used with --runs or --formats")
        config = EvaluationConfig(run_dirs=[str(args.matched.parent)], **{
            name: value
            for name in ("bootstrap_seed", "bootstrap_replicates")
            if (value := getattr(args, name)) is not None
        })
        print(evaluate_matched(args.matched, Path(args.output), config))
        return 0
    if args.bootstrap_seed is not None or args.bootstrap_replicates is not None:
        parser.error("Bootstrap overrides on the command line require --matched")
    settings = load_evaluation_config(args.config or DEFAULT_CONFIG).model_dump()
    for argument, key in (
        ("runs", "run_dirs"),
        ("output", "output_dir"),
        ("formats", "static_formats"),
    ):
        value = getattr(args, argument)
        if value is not None:
            settings[key] = value
    print(evaluate(EvaluationConfig.model_validate(settings)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Evaluation input/export error: {error}") from None
