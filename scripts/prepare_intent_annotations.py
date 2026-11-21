"""Create checked Sokoban intent annotations from local demonstrations."""

import argparse
import json
from pathlib import Path

from carl.icl.intent_annotations import prepare_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="JSON demonstration trajectories with fixed splits")
    parser.add_argument("--output", type=Path, required=True, help="New directory for D_A and its audit")
    parser.add_argument("--max-offset", type=int, default=8, help="Maximum primitive steps in a fragment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock-sketch", type=Path, help="Optional fixed mock response read verbatim from a local file")
    args = parser.parse_args()
    try:
        if args.mock_sketch is None:
            manifest = prepare_annotations(args.source, args.output, max_offset=args.max_offset, seed=args.seed)
        else:
            sketch = args.mock_sketch.read_text(encoding="utf-8")
            manifest = prepare_annotations(
                args.source, args.output, max_offset=args.max_offset, seed=args.seed,
                annotator=lambda _request: sketch, annotator_id="fixed-mock-sketch-v1",
            )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Annotation preparation failed: {error}\n")
    print(json.dumps({
        "output": str(args.output),
        **{name: manifest[name] for name in ("attempted", "accepted", "rejected", "unknown", "duplicates")},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
