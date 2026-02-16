#!/usr/bin/env python3

import argparse
import glob
from collections import Counter
from collections import defaultdict
from statistics import mean
from typing import Iterable

import joblib

from carl.planners.base import Experience
from carl.solver.nodes import SearchTreeNode


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(value) for value in values]


def _solving_path_nodes(solving_node: SearchTreeNode) -> list[SearchTreeNode]:
    nodes: list[SearchTreeNode] = []
    current: SearchTreeNode | None = solving_node
    while current is not None:
        nodes.append(current)
        current = current.parent_node
    nodes.reverse()
    return nodes


def _subgoal_positions_with_k(
    solving_node: SearchTreeNode,
    fallback_k_used: Iterable[int] | None,
) -> list[tuple[int, int | None]]:
    nodes = _solving_path_nodes(solving_node)
    if len(nodes) <= 1:
        return []

    fallback: list[int] | None = None
    if fallback_k_used is not None:
        fallback_list = list(fallback_k_used)
        if len(fallback_list) == len(nodes) - 1:
            fallback = fallback_list

    positions_with_k: list[tuple[int, int | None]] = []
    index = 0
    for idx, node in enumerate(nodes[1:]):
        low_level_path = node.low_level_path or []
        index += len(low_level_path)
        k_used = node.next_expand_with_k_generator
        if k_used is None and fallback is not None:
            k_used = fallback[idx]
        positions_with_k.append((index, k_used))
    return positions_with_k


def _action_path_length(exp: Experience) -> int | None:
    if exp.solution.solved and exp.solution.action_path is not None:
        return len(exp.solution.action_path)
    solving_node = exp.search_info.solving_node
    if solving_node is None:
        return None
    nodes = _solving_path_nodes(solving_node)
    return sum(len(node.low_level_path or []) for node in nodes[1:])


def _sliding_window_count(path_len: int, distance_range: list[int] | None) -> int:
    if path_len <= 1:
        return 0
    if distance_range is None:
        return path_len * (path_len - 1) // 2
    count = 0
    for position in range(path_len - 1):
        for dist in distance_range:
            inner_dist = min(dist, path_len - 1 - position)
            count += 1
            if position + inner_dist >= path_len - 1:
                break
    return count


def _k_offset_count(
    path_len: int,
    positions_with_k: list[tuple[int, int | None]],
    k: int,
    offsets: list[int],
) -> int:
    count = 0
    for position, k_used in positions_with_k:
        if k_used != k:
            continue
        if position < 0 or position >= path_len:
            continue
        for offset in offsets:
            start_index = position - k + offset
            if 0 <= start_index < position:
                count += 1
    return count


def _format_dict(values: dict[int, float], precision: int = 2) -> str:
    if not values:
        return "{}"
    items = [f"{key}:{value:.{precision}f}" for key, value in sorted(values.items())]
    return "{" + ", ".join(items) + "}"


def _format_int_dict(values: dict[int, int]) -> str:
    if not values:
        return "{}"
    items = [f"{key}:{value}" for key, value in sorted(values.items())]
    return "{" + ", ".join(items) + "}"


def _collect_experiences(files: list[str], max_experiences: int | None) -> list[tuple[str, list[Experience]]]:
    collected: list[tuple[str, list[Experience]]] = []
    for path in files:
        experiences = joblib.load(path)
        if not isinstance(experiences, list):
            continue
        if max_experiences is not None:
            experiences = experiences[:max_experiences]
        collected.append((path, experiences))
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generator target counts from solve_attempts joblib files.")
    parser.add_argument(
        "--glob",
        default="solve_attempts/*.joblib",
        help="Glob for solve_attempts joblib files.",
    )
    parser.add_argument(
        "--distance-range",
        default="",
        help="Comma-separated sliding window distances (e.g. 1,8,16).",
    )
    parser.add_argument(
        "--k-offsets",
        default="-1,1,-2,2,-3,3",
        help="Comma-separated k-offsets (e.g. -1,1,-2,2).",
    )
    parser.add_argument(
        "--k",
        default="",
        help="Comma-separated k values to summarize (default: infer from experiences).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process.",
    )
    parser.add_argument(
        "--max-experiences",
        type=int,
        default=None,
        help="Limit number of experiences per file.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f"No files matched {args.glob}")

    distance_range = _parse_int_list(args.distance_range) if args.distance_range else None
    offsets = _parse_int_list(args.k_offsets)
    requested_k = _parse_int_list(args.k) if args.k else []

    overall_experiences = 0
    overall_solved = 0
    overall_with_node = 0
    overall_path_lens: list[int] = []
    overall_sw_all = 0
    overall_sw_range = 0
    overall_k_totals: dict[int, int] = defaultdict(int)
    overall_k_exp_counts: dict[int, int] = defaultdict(int)

    collected = _collect_experiences(files, args.max_experiences)

    for path, experiences in collected:
        total = len(experiences)
        solved = [exp for exp in experiences if exp.solution.solved]
        solved_with_node = [exp for exp in solved if exp.search_info.solving_node is not None]
        path_lens: list[int] = []
        sw_all_total = 0
        sw_range_total = 0
        k_totals: dict[int, int] = defaultdict(int)
        k_exp_counts: dict[int, int] = defaultdict(int)

        for exp in solved_with_node:
            action_len = _action_path_length(exp)
            if action_len is None:
                continue
            path_len = action_len + 1
            path_lens.append(path_len)
            sw_all_total += _sliding_window_count(path_len, None)
            if distance_range is not None:
                sw_range_total += _sliding_window_count(path_len, distance_range)

            positions = _subgoal_positions_with_k(
                exp.search_info.solving_node, exp.solution.subgoal_distance_path
            )
            if requested_k:
                k_values = requested_k
            else:
                k_values = sorted({k for _, k in positions if k is not None})

            for k in k_values:
                if k is None:
                    continue
                k_totals[k] += _k_offset_count(path_len, positions, k, offsets)
                k_exp_counts[k] += 1

        avg_path_len = mean(path_lens) if path_lens else 0.0
        avg_sw_all = sw_all_total / len(path_lens) if path_lens else 0.0
        avg_sw_range = sw_range_total / len(path_lens) if path_lens and distance_range else 0.0
        avg_k = {k: (k_totals[k] / len(path_lens) if path_lens else 0.0) for k in k_totals}

        print(f"{path}")
        print(f"  experiences: {total}")
        print(f"  solved: {len(solved)}")
        print(f"  solved_with_node: {len(solved_with_node)}")
        if path_lens:
            print(
                f"  path_len: avg={avg_path_len:.2f}, min={min(path_lens)}, max={max(path_lens)}"
            )
        print(f"  sliding_window_all_pairs: total={sw_all_total}, avg={avg_sw_all:.2f}")
        if distance_range is not None:
            print(
                "  sliding_window_range"
                f" {distance_range}: total={sw_range_total}, avg={avg_sw_range:.2f}"
            )
        print(
            f"  k_offsets offsets={offsets}: totals={_format_int_dict(k_totals)}, "
            f"avg_per_solved={_format_dict(avg_k)}, exp_with_k={_format_int_dict(k_exp_counts)}"
        )

        overall_experiences += total
        overall_solved += len(solved)
        overall_with_node += len(solved_with_node)
        overall_path_lens.extend(path_lens)
        overall_sw_all += sw_all_total
        overall_sw_range += sw_range_total
        for k, value in k_totals.items():
            overall_k_totals[k] += value
            overall_k_exp_counts[k] += k_exp_counts[k]

    print("Overall")
    if overall_path_lens:
        overall_avg_path_len = mean(overall_path_lens)
        overall_avg_sw_all = overall_sw_all / len(overall_path_lens)
        overall_avg_sw_range = (
            overall_sw_range / len(overall_path_lens) if distance_range else 0.0
        )
        overall_avg_k = {
            k: (overall_k_totals[k] / len(overall_path_lens))
            for k in overall_k_totals
        }
        print(f"  experiences: {overall_experiences}")
        print(f"  solved: {overall_solved}")
        print(f"  solved_with_node: {overall_with_node}")
        print(
            f"  path_len: avg={overall_avg_path_len:.2f}, "
            f"min={min(overall_path_lens)}, max={max(overall_path_lens)}"
        )
        print(f"  sliding_window_all_pairs: total={overall_sw_all}, avg={overall_avg_sw_all:.2f}")
        if distance_range is not None:
            print(
                "  sliding_window_range"
                f" {distance_range}: total={overall_sw_range}, avg={overall_avg_sw_range:.2f}"
            )
        print(
            f"  k_offsets offsets={offsets}: totals={_format_int_dict(overall_k_totals)}, "
            f"avg_per_solved={_format_dict(overall_avg_k)}, "
            f"exp_with_k={_format_int_dict(overall_k_exp_counts)}"
        )


if __name__ == "__main__":
    main()
