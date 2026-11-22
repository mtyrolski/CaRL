"""Run a local CaRL CLLP reachability check on an exact pair of Sokoban boards."""

import argparse
import json
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.full_state_policy import FullStatePair, evaluate, load_settings


def main() -> int:
    """Load a local BERT or Qwen checkpoint and evaluate one full-state pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(load_settings(args.config), FullStatePair.model_validate_json(args.pair.read_text(encoding="utf-8")))
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        return parser.exit(1, f"Full-state evaluation failed: {error}\n")
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
