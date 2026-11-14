#!/usr/bin/env python3
"""Assess the frozen exploratory LoRA-controller addendum without GPU inference."""

import argparse
from pathlib import Path

from scripts.assess_fixed_controller import assess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Fresh output directory; defaults to the addendum assessment")
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--stop-epoch", type=float)
    args = parser.parse_args()
    run = args.run.resolve()
    args.plan = run / "policy_lora_addendum_plan.json"
    args.output = args.output or run / "controller_lora_addendum_assessment"
    args.data = args.inputs = args.results = args.predictions = args.matrix_plan = None
    assess(args)


if __name__ == "__main__":
    main()
