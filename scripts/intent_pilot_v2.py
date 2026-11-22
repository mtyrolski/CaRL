"""Freeze, meter and inspect the follow-up local intent diagnostic."""

import argparse
import json

from carl.icl.experiments.intent_pilot_budget import DOMAINS, PHASES
from carl.icl.experiments.intent_pilot_v2 import V1, prepare
from carl.icl.experiments.intent_pilot_v2_calibration import measure_context, prepare_calibration, publish_calibration
from carl.icl.experiments.intent_pilot_v2_run import PROMPTS, V2Ledger, diagnose, diagnostics_and_forecast, report, verify_trials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("prepare", help="Freeze new group descriptors and verify native witnesses")
    diagnostic = commands.add_parser("diagnostic", help="Run or resume the four frozen local Qwen3 requests")
    diagnostic.add_argument("--retry-censored", action="store_true", help="Retry wall cutoffs and infrastructure failures")
    commands.add_parser("prepare-calibration", help="Freeze both complete native-backed calibration pools")
    calibrate = commands.add_parser("calibrate", help="Measure one reserved calibration context")
    calibrate.add_argument("--context", type=int, required=True)
    commands.add_parser("calibration-report", help="Publish the calibration matrix and reconcile the ledger")
    commands.add_parser("report", help="Refresh per-attempt results and the v2 handoff")
    commands.add_parser("verify", help="Replay the saved native diagnostic attempts")
    commands.add_parser("status", help="Read the shared v2 allowance without launching a worker")
    run = commands.add_parser("run", help="Reserve and settle one local worker block")
    run.add_argument("--prompt", type=int, choices=PROMPTS, required=True)
    run.add_argument("--domain", choices=DOMAINS, default="both")
    run.add_argument("--phase", choices=PHASES, required=True)
    run.add_argument("--activity", required=True)
    run.add_argument("--seconds", type=int, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        ledger = V2Ledger(V1)
        if args.operation == "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            return int(ledger.run(args.prompt, args.domain, args.phase, args.activity,
                                  command, seconds=args.seconds, workers=args.workers))
        if args.operation == "prepare":
            result = prepare()
            diagnostics_and_forecast()
            report()
        elif args.operation == "diagnostic":
            diagnose(retry_censored=args.retry_censored)
            result = report()
        elif args.operation == "prepare-calibration":
            result = prepare_calibration()
            report()
        elif args.operation == "calibrate":
            result = measure_context(args.context)
        elif args.operation == "calibration-report":
            result = publish_calibration()
            report()
        elif args.operation == "report":
            result = report()
        elif args.operation == "verify":
            result = verify_trials()
        else:
            result = ledger.balance()
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, f"Intent pilot v2: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
