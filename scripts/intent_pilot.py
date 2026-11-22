"""Freeze intent-pilot splits and account for local experiment time."""

import argparse
import json
from pathlib import Path

from carl.icl.experiments.intent_pilot import freeze, verify
from carl.icl.experiments.intent_pilot_budget import DOMAINS, PHASES, PREPARATION, PilotLedger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("analysis/intent-pilot"))
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("init", help="Create or reopen the single persistent ledger")
    commands.add_parser("freeze", help="Freeze whole episodes and bind them to the campaign")
    commands.add_parser("verify", help="Check split hashes and display the safe resume summary")
    commands.add_parser("status", help="Read the budget and preparation rows")
    replay = commands.add_parser("replay", help="Check checkpoint replay on a public integration fixture")
    replay.add_argument("--domain", required=True, choices=("sokoban", "craftax"))
    replay.add_argument("--output", type=Path, required=True)
    data = commands.add_parser("annotations", help="Collect, rebuild or verify the checked pilot datasets")
    data.add_argument("operation", choices=("collect", "build", "verify"))
    data.add_argument("--domain", choices=("sokoban", "craftax"), required=True)
    direct = commands.add_parser("direct", help="Run the matched real-local direct baseline")
    direct.add_argument("operation", choices=("freeze", "run"))
    direct.add_argument("--controller", type=Path)
    direct.add_argument("--output", type=Path)
    direct.add_argument("--resume", action="store_true", help="Continue the missing starts of a partial direct report")
    hindsight = commands.add_parser("hindsight", help="Collect assigned intents and freeze checked hindsight support")
    hindsight.add_argument("operation", choices=("plan", "collect", "packets", "freeze", "verify"))
    run = commands.add_parser("run", help="Reserve worker time and execute a local command")
    run.add_argument("--prompt", required=True, choices=range(1, 10), type=int, metavar="1..9")
    run.add_argument("--domain", default="both", choices=DOMAINS)
    run.add_argument("--phase", required=True, choices=PHASES)
    run.add_argument("--activity", required=True)
    run.add_argument("--seconds", required=True, type=int, help="Deadline per worker, excluding cleanup headroom")
    run.add_argument("--workers", type=int, default=1, help="Maximum number of independent concurrent workers")
    run.add_argument("command", nargs=argparse.REMAINDER)
    prep = commands.add_parser("record-preparation", help="Track procurement, annotation time or Codex usage")
    prep.add_argument("--prompt", required=True, choices=range(1, 10), type=int, metavar="1..9")
    prep.add_argument("--activity", required=True, choices=PREPARATION)
    prep.add_argument("--status", required=True)
    prep.add_argument("--seconds", type=float)
    prep.add_argument("--input-tokens", type=int)
    prep.add_argument("--output-tokens", type=int)
    prep.add_argument("--note", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        ledger = PilotLedger(root, create=args.action == "init")
        if args.action == "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            return ledger.run(args.prompt, args.domain, args.phase, args.activity,
                              command, seconds=args.seconds, workers=args.workers)
        if args.action == "replay":
            from carl.icl.experiments.intent_pilot_replay import run_replay

            run_replay(args.domain, args.output)
            print(args.output)
            return 0
        if args.action == "annotations":
            from carl.icl.experiments import intent_pilot_annotations as annotations

            if verify(root)["split_sha256"] != ledger.split_digest():
                raise ValueError("The split manifest does not match the campaign ledger")
            if args.operation == "collect":
                result = annotations.collect(root, args.domain)
            else:
                result = annotations.build(root, args.domain, verify_only=args.operation == "verify")
            print(json.dumps(result, indent=2))
            return 0
        if args.action == "direct":
            from carl.icl.experiments.intent_pilot_direct import freeze_validation, run_validation

            if verify(root)["split_sha256"] != ledger.split_digest():
                raise ValueError("The split manifest does not match the campaign ledger")
            if args.operation == "freeze":
                if args.resume:
                    raise ValueError("--resume is available for direct run")
                result = freeze_validation(root)
            else:
                if args.controller is None or args.output is None:
                    raise ValueError("Direct run requires --controller and --output")
                result = run_validation(root, args.controller, args.output, resume=args.resume)
            print(json.dumps(result, indent=2))
            return 0
        if args.action == "hindsight":
            split = verify(root)["split_sha256"]
            if split != ledger.split_digest():
                raise ValueError("The split manifest does not match the campaign ledger")
            if args.operation in ("plan", "collect"):
                from carl.icl.experiments.intent_pilot_attempts import collect, plan

                result = plan(root, split) if args.operation == "plan" else collect(root, split)
            else:
                from carl.icl.experiments.intent_pilot_hindsight import build, packets

                result = packets(root, split) if args.operation == "packets" else build(root, split, args.operation == "verify")
                if args.operation == "freeze":
                    ledger.bind_candidate_pool(result["pool_version"], result["pool_sha256"])
                elif args.operation == "verify" and ledger.candidate_pool_digest(result["pool_version"]) != result["pool_sha256"]:
                    raise ValueError("The candidate pool does not match its campaign ledger binding")
            print(json.dumps(result, indent=2))
            return 0
        if args.action == "record-preparation":
            ledger.record_preparation(
                args.prompt, args.activity, args.status, args.note, seconds=args.seconds,
                input_tokens=args.input_tokens, output_tokens=args.output_tokens,
            )
        if args.action == "freeze":
            freeze(root)
            ledger.bind_split(verify(root)["split_sha256"])
        if args.action in ("freeze", "verify"):
            summary = verify(root)
            if summary["split_sha256"] != ledger.split_digest():
                raise ValueError("The split manifest does not match the campaign ledger")
            summary["budget"] = {key: value for key, value in ledger.status().items()
                                 if key not in {"jobs", "separate_preparation"}}
        else:
            summary = ledger.status()
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, f"Intent pilot: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
