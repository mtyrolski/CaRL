"""Prepare, inspect or explicitly execute the measured 100-board launch."""

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from filelock import FileLock

from scripts.launch100.forecast import forecast
from scripts.launch100.plan import ASSETS, OUTPUT, build_plan, ledger_status, load_assets, prepare
from scripts.launch100.runtime import execute, status


def main() -> None:
    """Default to an API-free plan; execution always requires a prepared manifest and USD cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preset", choices=("core", "extended"), default="core")
    parser.add_argument("--action-budget", type=int, default=4000)
    parser.add_argument("--usd-cap", type=float, help="Absolute shared campaign USD ceiling, maximum 40")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.execute and args.usd_cap is None:
        parser.error("--execute requires --usd-cap (absolute shared campaign ceiling)")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    manifest = args.output / "manifest.json"
    if args.status:
        if not manifest.exists():
            parser.error("No prepared manifest; use --prepare first")
        print(json.dumps(status(json.loads(manifest.read_text()), args.output, write_csv=True), indent=2))
        return
    methods = ["relative_u", "absolute_d"]
    if args.preset == "extended":
        methods += ["fixed_k", "verbal_relative"]
    with redirect_stdout(io.StringIO()):
        bundle = load_assets(args.assets)
        plan = build_plan(bundle, methods, args.action_budget)
    if not args.prepare and not args.execute:
        summary = {key: value for key, value in plan.items()
                   if key not in {"configs", "source_hashes", "cohort", "prompt_previews"}}
        summary.update(family="real_12x12x4", board_ids=[row["id"] for row in plan["cohort"]],
                       ledger=ledger_status(), forecast=forecast(plan))
        print(json.dumps(summary, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output) + ".lock", timeout=0):
        if args.prepare:
            prepare(plan, args.output)
            print(json.dumps(status(plan, args.output, write_csv=True), indent=2))
            return
        if not manifest.exists() or json.loads(manifest.read_text()) != plan:
            raise ValueError("Missing or changed frozen plan; prepare a new output directory before execution")
        execute(bundle, plan, args.output, args.usd_cap, resume=args.resume)


if __name__ == "__main__":
    main()
