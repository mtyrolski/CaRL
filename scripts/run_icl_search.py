"""Prepare, calibrate and evaluate in-context explicit-state search."""
import argparse
from pathlib import Path
from typing import Final

from carl.icl.config import ROOT, ExperimentConfig, load_config
from carl.icl.datasets import prepare, prepare_value_examples
from carl.icl.experiment import (
    matrix_configs,
    output_directory,
    preview,
    run_calibration,
    run_condition,
    run_matrix,
    tune,
)


DEFAULT_CONFIG: Final[str] = "configs/icl/experiments/relative.json"
DEFAULT_MATRIX: Final[str] = "configs/icl/experiments/matrix_pilot.json"
COMMANDS: Final = ("prepare", "prepare-value", "preview", "calibrate", "run", "matrix", "tune")


def build_parser() -> argparse.ArgumentParser:
    """Expose explicit offline/live modes and validated experiment configuration paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=COMMANDS,
        nargs="?",
        default="preview",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--matrix", default=DEFAULT_MATRIX)
    parser.add_argument(
        "--model",
        help="Override the fallback model used by roles without an explicit model",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path,
        help="Share or restore a local query cache (relative to the CaRL checkout)",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Use the configured APIs; may incur charges")
    modes.add_argument(
        "--mock", action="store_true", help="Heuristic offline plumbing check, not a Luna benchmark"
    )
    modes.add_argument(
        "--replay", action="store_true", help="Use only previously cached live responses"
    )
    return parser


def main() -> int:
    """Dispatch one experiment operation and return a meaningful process exit status."""
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.cache_dir:
        overrides["cache_dir"] = str(args.cache_dir)
    if overrides:
        config = ExperimentConfig.model_validate({**config.model_dump(), **overrides})
    provider = "openai" if args.execute else "mock" if args.mock else "replay"
    if args.command == "prepare":
        manifest = prepare(config)
        print(
            f"Prepared {manifest['total_unique_layouts']} disjoint verified layouts: {config.path('data_dir')}"
        )
        return 0
    if args.command == "prepare-value":
        count = prepare_value_examples(config)
        print(f"Prepared {count} verified value examples in {config.path('data_dir')}")
        return 0
    if args.command == "preview":
        print(preview(config, provider))
        return 0
    if not (args.execute or args.mock or args.replay):
        parser.error("Choose --execute, --mock, or --replay explicitly for inference")
    if args.command == "calibrate":
        result = run_calibration(config, provider)
        contexts = {example["context"] for example in result["examples"]}
        return 0 if result["status"] == "completed" and len(contexts) >= config.few_shot else 1
    directory = args.output or output_directory(config, provider, args.command)
    if args.command == "run":
        rows = run_condition(config, provider, directory)
        if any(row["finish_reason"] == "api_error" for row in rows):
            return 1
    elif args.command == "matrix":
        matrix = Path(args.matrix)
        configs = matrix_configs(matrix if matrix.is_absolute() else ROOT / matrix)
        if overrides:
            configs = [
                ExperimentConfig.model_validate({**c.model_dump(), **overrides})
                for c in configs
            ]
        run_matrix(configs, provider, directory)
    else:
        tune(config, provider, directory)
    print(f"Results: {directory}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Configuration/data error: {error}") from None
