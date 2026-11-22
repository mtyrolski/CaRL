"""Freeze, preview and execute predicate policies and planners in Sokoban."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.config import load_config
from carl.icl.intent_domain import decode_intent
from carl.icl.intents.generation.data import parse_json
from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.planning.craftax_experiment import add_arguments as add_craftax_arguments
from carl.icl.planning.craftax_experiment import run as run_craftax
from carl.icl.planning.intent_search import SearchMode
from carl.icl.planning.intent_value import fit_value
from carl.icl.predicate_icl import MockResponses, PredicateCondition, PredicateQuery
from carl.icl.types import JSONObject


def _read(path: Path) -> object:
    return parse_json(path.read_text(encoding="utf-8"))


def _emit(result: JSONObject, path: Path | None) -> None:
    if path is None:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        write_json(path, result)
        print(path)


def _execute(args: argparse.Namespace) -> int:
    if args.command == "craftax":
        return run_craftax(args)
    if args.command == "fit-value":
        fitted = fit_value(args.source, discount=args.discount)
        fitted.save(args.output)
        _emit(fitted.identity(), None)
        return 0
    condition = PredicateCondition(load_config(args.config), args.data, args.manifest, args.policy_data, args.policy_audit)
    if args.command == "freeze":
        support = condition.freeze()
        _emit({"status": "frozen", "support_fingerprint": support.fingerprint}, args.output)
        return 0
    query = PredicateQuery.model_validate(_read(args.query))
    if args.command == "preview":
        _emit(condition.preview(query, decode_intent(_read(args.intent))), args.output)
        return 0
    planning_mode = SearchMode(args.planning_mode) if args.command == "search" else None
    mock = None if args.mock_response is None else MockResponses.model_validate(_read(args.mock_response))
    result = condition.infer(
        query, mode=args.mode, planning_mode=planning_mode, mock=mock, validation_result=args.validation_result,
    )
    _emit(result, args.output)
    success_status = "solved" if planning_mode is not None else RolloutStatus.ACHIEVED
    return 0 if result["status"] == success_status else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Expose explicit inference modes and a keyless preview of both roles."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    add_craftax_arguments(actions.add_parser("craftax", help="Compare four planners on a pinned Craftax task"))
    fit = actions.add_parser("fit-value", help="Fit V from replay-verified training demonstrations")
    fit.add_argument("--source", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--discount", type=float, default=0.99)
    for name in ("freeze", "preview", "run", "search"):
        action = actions.add_parser(name)
        action.add_argument("--config", type=Path, required=True)
        action.add_argument("--data", type=Path, required=True, help="Checked D_A.jsonl or D_C.jsonl for G")
        action.add_argument("--manifest", type=Path, required=True, help="Frozen support manifest")
        action.add_argument("--policy-data", type=Path, help="Separate checked D_A for π")
        action.add_argument("--policy-audit", type=Path, help="Override the adjacent annotation_audit.jsonl for π")
        action.add_argument("--output", type=Path)
        if name != "freeze":
            action.add_argument("--query", type=Path, required=True, help="Visible query JSON and held-out split metadata")
        if name == "preview":
            action.add_argument("--intent", type=Path, required=True, help="Proposed intent for the policy preview")
            action.set_defaults(mode=None)
        if name == "search":
            action.add_argument("--planning-mode", choices=tuple(mode for mode in SearchMode if mode is not SearchMode.PLAN_AND_ACT), required=True)
        if name in ("run", "search"):
            action.add_argument("--mode", choices=("mock", "replay", "openai", "local"), required=True,
                                help="Use local weights for local roles and this mode for any API roles")
            action.add_argument("--mock-response", type=Path, help="Fixture JSON required with --mode mock")
            action.add_argument("--validation-result", type=Path, help="Successful validation output for test inference")
    args = parser.parse_args(argv)
    if args.command in ("run", "search") and (args.mode == "mock") != (args.mock_response is not None):
        parser.error("--mock-response is required in mock mode and unavailable in replay or live mode")
    try:
        return _execute(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        return parser.exit(1, f"Predicate ICL failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
