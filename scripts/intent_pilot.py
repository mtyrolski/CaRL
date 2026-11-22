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
