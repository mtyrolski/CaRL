"""Plan or run a matched per-budget CaRL/Luna solver sweep."""
import argparse
import json
from pathlib import Path
from typing import Final

from carl.icl.budget_sweep import describe_plan, load_sweep, run_sweep
from carl.icl.config import ROOT, ExperimentConfig

DEFAULT_CONFIG: Final[str] = "configs/icl/budget_sweeps/pilot.json"


def main() -> int:
    """Require a provider or an explicit offline plan; never infer live execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", help="Override the output directory")
    parser.add_argument("--cache-dir", help="Use a shared or restored local query cache")
    parser.add_argument(
        "--resume", action="store_true", help="Resume a matching existing sweep"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--plan", action="store_true", help="Show the trial count; do not infer"
    )
    modes.add_argument("--mock", action="store_true", help="Offline interface check")
    modes.add_argument(
        "--replay",
        action="store_true",
        help="Read only previously cached live responses",
    )
    modes.add_argument(
        "--execute", action="store_true", help="Use OpenAI; incurs charges"
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser()
    config_path = (config_path if config_path.is_absolute() else ROOT / config_path).resolve()
    if args.execute and config_path == (ROOT / DEFAULT_CONFIG).resolve():
        parser.error("The default configuration cannot run live; supply an explicit experiment configuration")
    spec, configs = load_sweep(args.config)
    if args.cache_dir:
        configs = [
            ExperimentConfig.model_validate({**config.model_dump(), "cache_dir": args.cache_dir})
            for config in configs
        ]
    plan = describe_plan(spec, configs)
    if args.output:
        plan["output_dir"] = args.output
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan:
        return 0
    provider = "openai" if args.execute else "mock" if args.mock else "replay"
    path = Path(args.output or spec.output_dir).expanduser()
    run_sweep(
        spec,
        configs,
        provider,
        path if path.is_absolute() else ROOT / path,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Sweep error: {error}") from None
