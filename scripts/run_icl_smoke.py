"""Prepare, preview and evaluate a small matched Sokoban predicate sweep offline."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from carl.icl.experiments.sokoban_smoke import DEFAULT_CONFIG, run


def main(argv: Sequence[str] | None = None) -> int:
    """Run without credentials against checked fixture annotations and production parsers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--profile", help="Override the default declared profile")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after preparing, previewing and checking the selected profile")
    args = parser.parse_args(argv)
    try:
        print(run(args.config, args.output, profile=args.profile, prepare_only=args.prepare_only))
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.exit(1, f"Sokoban smoke failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
