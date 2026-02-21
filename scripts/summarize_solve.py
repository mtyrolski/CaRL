#!/usr/bin/env python3

import argparse
import glob
from collections import defaultdict
from statistics import mean
from typing import Iterable, cast
from tqdm import tqdm
import joblib

from carl.planners.base import Experience
from carl.solver.nodes import SearchTreeNode
import os

def extract_ks_from_experiences(experience_path: str) -> str:
    base_name = os.path.basename(experience_path)
    return base_name.split('solved_problems_')[-1].split('_')[0]

def summarize(paths: str) -> None:
    pass

def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generator target counts from solve_attempts joblib files.")
    parser.add_argument(
        "--glob",
        default="solve_attempts/*.joblib",
        help="Glob for solve_attempts joblib files.",
    )

    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob}")

    uq_keys = sorted(list(map(extract_ks_from_experiences, files)), key=lambda x: int(x[1:]))
    experiences_by_solver: dict[str, list[Experience]] = {k: [] for k in uq_keys}
    
    for path in tqdm(files, desc="Processing experience files", unit="file", total=len(files)):
        k = extract_ks_from_experiences(path)
        experiences = joblib.load(path)
        experiences_by_solver[k].extend(experiences)
    
    for k, experiences in experiences_by_solver.items():
        total = len(experiences)
        solved = sum(1 for exp in experiences if exp.solution.solved)
        print(f"K={k}: Solved {solved}/{total} ({solved/total:.2%})")
    

    

if __name__ == "__main__":
    main()
