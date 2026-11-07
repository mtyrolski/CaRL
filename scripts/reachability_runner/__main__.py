"""Frozen reachability trials; live requires an explicitly authorised campaign."""
import argparse
import json
from pathlib import Path

from carl.icl.reachability_data import verify_dataset
from scripts.reachability_runner.engine import absolute, contract, export, load_study, plan, prepare, run, status, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "preflight", "prepare", "run", "resume", "status", "verify", "export"))
    parser.add_argument("--config", type=Path, default=Path("configs/icl/reachability_runner/offline_smoke.json"))
    parser.add_argument("--max-trials", type=int, help="Bound new trials in this invocation")
    parser.add_argument("--interrupt-after-actions", type=int, help="Simulate an interruption for offline testing")
    args = parser.parse_args()
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    if args.interrupt_after_actions is not None and args.interrupt_after_actions < 1:
        parser.error("--interrupt-after-actions must be positive")
    study = load_study(args.config)
    if args.command == "plan":
        result = plan(study)
    elif args.command == "preflight":
        report = verify_dataset(absolute(study.dataset_dir))
        frozen = contract(study)
        result = {"dataset": "verified", "classification": report["classification"], "mode": study.mode,
                  "contract_id": frozen["contract_id"], "trial_count": len(frozen["jobs"]),
                  "profiles": {arm: {"model": p.model_id, "provider": p.provider, "endpoint": p.endpoint,
                                      "reasoning": p.reasoning, "format": p.response_format,
                                      "pricing_explicit": p.pricing is not None} for arm, p in study.arms.items()},
                  "live_ready": False,
                  "note": "No API call or credential check. Live requires a separately authorised campaign, "
                          "positive limits and an explicit model profile; generation access remains unverified."}
    elif args.command == "prepare":
        result = {"prepared": True, "contract_id": prepare(study)["contract_id"]}
    elif args.command in {"run", "resume"}:
        result = run(study, resume=args.command == "resume", max_trials=args.max_trials,
                     interrupt_after_actions=args.interrupt_after_actions)
    elif args.command == "status":
        result = status(study)
    elif args.command == "verify":
        result = verify(study)
    else:
        result = {"exported": True, "summary": export(study)["status"]}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
