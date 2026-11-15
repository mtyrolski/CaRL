#!/usr/bin/env python3
"""Build the overnight Sokoban campaign report from saved, local artifacts.

This module performs no model inference and does not start or resume jobs.
Unknown infrastructure outcomes remain unknown, including in plotted lower
bounds. Stage-five estimates always group by the original canonical layout.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import math
import os
import random
import shlex
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from carl.icl.intent_assessment import bootstrap
from scripts.report_overnight_stage03 import atomic_text
from scripts.report_overnight_stage03 import read_csv
from scripts.report_overnight_stage03 import read_json
from scripts.report_overnight_stage03 import read_jsonl
from scripts.report_overnight_stage03 import write_csv

Record = dict[str, Any]
FAMILIES = ('12x12x4', '12x12x10')
METHODS = ('adasubs', 'ksubs', 'direct_policy')
METHOD_LABELS = {'adasubs': 'AdaSubS', 'ksubs': 'Fixed-k kSubS',
                 'direct_policy': 'Direct policy', 'primitive_search': 'Primitive search',
                 'primitive_astar': 'Primitive A* (secondary)'}
COLOURS = {'adasubs': '#22628b', 'ksubs': '#bb6129', 'direct_policy': '#47834f',
           'primitive_search': '#865a9b', 'primitive_astar': '#865a9b'}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def optional_json(path: Path) -> Record | None:
    if not path.is_file():
        return None
    result = read_json(path)
    if not isinstance(result, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return result


def optional_csv(path: Path) -> list[Record]:
    return read_csv(path) if path.is_file() else []


def checked_shard_files(shard_path: Path, manifest_path: Path) -> tuple[list[Path], Record]:
    """Verify the atomic merge manifest; retain partial work with explicit omissions."""
    available = sorted(shard_path.rglob('*.json')) if shard_path.is_dir() else []
    if not manifest_path.is_file():
        return available, {'state': 'missing', 'hash_verified': False, 'rejected': [], 'unlisted': [],
                           'source_unavailable': []}
    try:
        manifest = read_json(manifest_path)
        hashes, sources = manifest['shards_sha256'], manifest.get('source_paths', {})
        if not isinstance(hashes, dict) or not isinstance(sources, dict):
            raise ValueError('expected shard-hash and source-path objects')
        if manifest.get('state') not in ('complete', 'partial') or manifest.get('merged_jobs') != len(hashes):
            raise ValueError('invalid merge state or merged-job count')
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return [], {'state': 'invalid', 'hash_verified': False, 'rejected': [str(error)], 'unlisted': [],
                    'source_unavailable': []}
    accepted, rejected, source_unavailable = [], [], []
    for name, expected in sorted(hashes.items()):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not isinstance(expected, str):
            rejected.append(f'Invalid relative shard path or hash: {name}')
            continue
        path = shard_path / relative
        if not path.is_file() or sha256(path) != expected:
            rejected.append(f'Missing or changed merged shard: {name}')
            continue
        original = sources.get(name)
        if original:
            if not Path(original).is_file():
                source_unavailable.append(name)
            elif sha256(Path(original)) != expected:
                rejected.append(f'Merged shard differs from its original: {name}')
                continue
        accepted.append(path)
    listed = set(hashes)
    unlisted = [str(path.relative_to(shard_path)) for path in available
                if str(path.relative_to(shard_path)) not in listed]
    return accepted, {'state': manifest['state'], 'hash_verified': not rejected and not unlisted,
                      'planned_jobs': manifest.get('planned_jobs'), 'merged_jobs': manifest['merged_jobs'],
                      'rejected': rejected, 'unlisted': unlisted, 'source_unavailable': source_unavailable}


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def integer(value: Any) -> int | None:
    value = numeric(value)
    if value is None or not value.is_integer():
        return None
    return int(value)


def dataset_summary(campaign: Path) -> Record:
    """Extract a few comparable prior results from their saved summaries."""
    status = optional_json(campaign / 'stage_01/assessment/assessment.json') or {}
    first = optional_csv(campaign / 'stage_01/assessment/summary.csv')
    selected = []
    for row in first:
        if (row.get('source_family') == 'all' and row.get('model_id') in ('cllp8', 'cllp32')
                and row.get('k') in ('16', '32') and row.get('budget') == '32'):
            selected.append({key: row[key] for key in ('family', 'model_id', 'k', 'budget',
                                                        'hits', 'planned', 'complete', 'missing')})
    rank = optional_csv(campaign / 'stage_02/analysis_evaluation/rank_summary.csv')
    scale = []
    for row in rank:
        if (row.get('scope') == 'family' and row.get('source_family') == 'all'
                and row.get('controller') in ('cllp8', 'cllp32') and row.get('L') == '32'):
            scale.append({key: row[key] for key in ('family', 'controller', 'L', 'contexts_with_two_distinct_d',
                                                     'error_contexts', 'mean_abs_u_difference',
                                                     'strict_comparable', 'inversions')})
    second = optional_json(campaign / 'stage_02_status.json') or {}
    third = optional_json(campaign / 'stage_03_summary.json') or {}
    fourth = optional_json(campaign / 'stage_04_summary.json') or {}
    fresh = third.get('fresh_generation_status') or {}
    third_rows = third.get('fresh_summary') or []
    grounding = [row for row in third_rows if row.get('family') == '12x12x4'
                 and row.get('source_family') == 'rl_validation_12x12x4'
                 and row.get('method') == 'checkpoint220' and row.get('condition') in ('original', 'd4')]
    icl = [row for row in third_rows if row.get('family') == '12x12x10'
           and row.get('method') == 'in_context' and row.get('condition') == 'original']
    figure = [row for row in fourth.get('figure', []) if row.get('transform') == 'identity'
              and row.get('metric') == 'valid']
    return {
        'stage_01': {'planned_trials': status.get('planned_model_trials'),
                     'completed_trials': status.get('completed_model_trials'),
                     'missing_trials': status.get('missing_model_trials'), 'selected': selected},
        'stage_02': {'evaluation_attempts': second.get('evaluation_attempts'),
                     'train_attempts': second.get('train_attempts'), 'conditional_rank': scale},
        'stage_03': {'overlap': third.get('overlap'), 'fresh_generation_status': fresh,
                     'x4_checkpoint_transforms': grounding, 'x10_icl_original': icl,
                     'fresh_generation_expected': 432, 'fresh_boards_expected': 48},
        'stage_04': {'dataset': fourth.get('dataset'), 'evaluation': fourth.get('evaluation'),
                     'identity_figure': figure, 'repetition_family': fourth.get('repetition_family'),
                     'predicate_coverage': fourth.get('predicate_coverage'),
                     'paired_family': fourth.get('paired_family')},
    }


def methods_from_freeze(freeze: Record) -> list[str]:
    configured = freeze.get('methods', list(METHODS))
    if not isinstance(configured, list) or not configured or any(not isinstance(x, str) for x in configured):
        raise ValueError('Freeze must contain a nonempty method-name list')
    if len(configured) != len(set(configured)):
        raise ValueError('Freeze contains duplicate methods')
    return configured


def initial_board_inventory(boards: list[Record]) -> list[Record]:
    """Verify start-board object counts and summarise initially occupied goals."""
    grouped: dict[tuple[str, str], list[Record]] = {}
    for board in boards:
        if board.get('split') == 'evaluation':
            grouped.setdefault((board['family'], board['source_family']), []).append(board)
    result = []
    for (family, source), group in sorted(grouped.items()):
        expected = {'12x12x4': 4, '12x12x10': 10}[family]
        distribution: Counter[str] = Counter()
        unknown = 0
        for board in group:
            rows = board.get('initial_state')
            if rows is None:
                unknown += 1
                continue
            if (not isinstance(rows, list) or len(rows) != 12
                    or any(not isinstance(row, str) or len(row) != 12 or set(row) - set('#-.*$@+') for row in rows)):
                raise ValueError(f"Invalid full initial board: {board.get('layout_id')}")
            cells = ''.join(rows)
            boxes, goals, players = (cells.count('$') + cells.count('*'),
                                     cells.count('.') + cells.count('*') + cells.count('+'),
                                     cells.count('@') + cells.count('+'))
            occupied = cells.count('*')
            if boxes != expected or goals != expected or players != 1:
                raise ValueError(f"Incorrect box, goal or player count: {board.get('layout_id')}")
            declared = board.get('initial_boxes_on_goals')
            if declared is not None and integer(declared) != occupied:
                raise ValueError(f"Incorrect initial goal count: {board.get('layout_id')}")
            solved = board.get('initially_solved')
            if solved is not None and solved != (occupied == expected):
                raise ValueError(f"Incorrect initial solved flag: {board.get('layout_id')}")
            distribution[str(occupied)] += 1
        result.append({'family': family, 'source_family': source, 'planned_boards': len(group),
                       'checked_initial_states': len(group) - unknown, 'unknown_initial_states': unknown,
                       'initial_boxes_on_goals_distribution': dict(sorted(distribution.items(), key=lambda x: int(x[0])))})
    return result


def planned_cases(boards: list[Record], freeze: Record, shards: list[Record]) -> list[Record]:
    """Join all frozen evaluation layouts to all methods without dropping gaps."""
    methods = methods_from_freeze(freeze)
    evaluation = [board for board in boards if board.get('split') == 'evaluation']
    by_layout = {row.get('layout_id'): row for row in evaluation}
    if (not evaluation or None in by_layout or len(by_layout) != len(evaluation)
            or any(row.get('family') not in FAMILIES or not row.get('source_family') for row in evaluation)):
        raise ValueError('Evaluation boards must have distinct canonical layout IDs and known families/sources')
    observed: dict[tuple[str, str], Record] = {}
    for shard in shards:
        key = (shard.get('layout_id'), shard.get('method'))
        if key[0] not in by_layout or key[1] not in methods:
            raise ValueError(f'Unplanned board/method shard: {key}')
        if key in observed:
            raise ValueError(f'Duplicate board/method shard: {key}')
        board = by_layout[key[0]]
        if any(shard.get(field) != board.get(field) for field in ('family', 'source_family', 'split')):
            raise ValueError(f'Shard metadata differs from its frozen board: {key}')
        validate_result(shard, freeze)
        observed[key] = shard
    return [{**{key: board.get(key) for key in ('layout_id', 'family', 'source_family', 'split', 'board_rank')},
             'method': method, 'result': observed.get((board['layout_id'], method))}
            for board in evaluation for method in methods]


def checked_cases(boards: list[Record], freeze: Record, shard_files: list[Path]) -> tuple[list[Record], list[str]]:
    """Retain valid shards while reporting malformed or conflicting cases as missing."""
    planned_cases(boards, freeze, [])  # Fail early if the board plan itself is inconsistent.
    accepted: dict[tuple[str, str], tuple[Path, Record]] = {}
    conflicts: set[tuple[str, str]] = set()
    invalid = []
    for path in shard_files:
        try:
            row = read_json(path)
            if not isinstance(row, dict):
                raise ValueError('shard must contain a JSON object')
            planned_cases(boards, freeze, [row])
            key = (row['layout_id'], row['method'])
            if key in conflicts:
                raise ValueError(f'additional conflicting shard for {key}')
            if key in accepted:
                previous, _ = accepted.pop(key)
                conflicts.add(key)
                invalid.append(f'{previous}: duplicate board/method {key}')
                raise ValueError(f'duplicate board/method {key}')
            accepted[key] = path, row
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            invalid.append(f'{path}: {type(error).__name__}: {error}')
    return planned_cases(boards, freeze, [row for _, row in accepted.values()]), invalid


def validate_result(row: Record, freeze: Record) -> None:
    status = row.get('status')
    if status not in ('completed', 'timeout', 'infra_error'):
        raise ValueError(f'Unknown stage-five job status: {status!r}')
    solved = row.get('solved')
    if status == 'infra_error':
        if solved is not None:
            raise ValueError('Infrastructure errors have unknown solved status')
        return
    if not isinstance(solved, bool) or (status == 'timeout' and solved):
        raise ValueError('Completed/timed-out jobs require consistent Boolean solved status')
    transitions = integer(row.get('transitions'))
    limit = integer(freeze.get('transition_budget'))
    if transitions is None or transitions < 0 or (limit is not None and transitions > limit):
        raise ValueError('Invalid or over-budget simulator transition count')
    if solved:
        reached = integer(row.get('solve_at_transition'))
        length = integer(row.get('solution_length'))
        if reached is None or not 0 <= reached <= transitions or length is None or not 0 <= length <= reached:
            raise ValueError('Solved record lacks a valid transition count and plan length')
        if row.get('solution_verified') is not True:
            raise ValueError('Solved record lacks successful environment verification')
        elapsed = numeric(row.get('solve_wall_seconds'))
        if elapsed is None:
            elapsed = numeric(row.get('wall_seconds'))
        if elapsed is None or elapsed < 0:
            raise ValueError('Solved record lacks an observed wall time')


def known(case: Record) -> bool:
    result = case['result']
    return bool(result is not None and result.get('status') in ('completed', 'timeout')
                and isinstance(result.get('solved'), bool))


def joint_success(case: Record, freeze: Record) -> bool:
    """Require the entire plan within both the transition and wall caps."""
    result = case['result']
    if not known(case) or not result['solved']:
        return False
    transitions = numeric(result.get('solve_at_transition'))
    seconds = numeric(result.get('solve_wall_seconds'))
    if seconds is None:
        seconds = numeric(result.get('wall_seconds'))
    transition_cap = numeric(freeze.get('transition_budget'))
    wall_cap = numeric(freeze.get('per_job_wall_seconds'))
    return (transitions is not None and seconds is not None and transition_cap is not None and wall_cap is not None
            and transitions <= transition_cap and seconds <= wall_cap)


def at_budget(case: Record, axis: str, threshold: float, freeze: Record) -> bool:
    result = case['result']
    if not joint_success(case, freeze):
        return False
    if axis == 'transitions':
        cost = numeric(result.get('solve_at_transition'))
    elif axis == 'wall_seconds':
        cost = numeric(result.get('solve_wall_seconds'))
        if cost is None:
            cost = numeric(result.get('wall_seconds'))
    else:
        raise ValueError(f'Unknown budget axis: {axis}')
    return cost is not None and cost <= threshold


def known_sum(rows: list[Record], field: str) -> int | float | None:
    values = [numeric(row.get(field)) for row in rows]
    if not values or any(value is None for value in values):
        return None
    total = sum(values)
    return int(total) if total.is_integer() else total


def input_token_sum(rows: list[Record], field: str, call_field: str, tokens_per_call: int) -> int | None:
    """Use explicit token counts, or the verified fixed tokenizer length per call."""
    if not rows:
        return None
    values = []
    for row in rows:
        value, calls = integer(row.get(field)), integer(row.get(call_field))
        if value is None and calls is not None:
            value = calls * tokens_per_call
        if value is None:
            return None
        values.append(value)
    return sum(values)


def known_mapping_total(rows: list[Record], field: str) -> dict[str, int] | None:
    mappings = [row.get(field) for row in rows]
    if not mappings or any(not isinstance(item, dict) for item in mappings):
        return None
    total: Counter[str] = Counter()
    for item in mappings:
        for key, value in item.items():
            count = integer(value)
            if count is None or count < 0:
                raise ValueError(f'Invalid {field} count: {key}={value!r}')
            total[str(key)] += count
    return dict(sorted(total.items()))


def generation_horizons(rows: list[Record]) -> Record:
    """Count actual generator calls by logged k, and flag incomplete traces."""
    if not rows:
        return {'calls_by_k': None, 'boards_using_k': None, 'boards_using_k1_or_k4': None,
                'trace_events': None, 'reported_calls': None, 'missing_traces': None, 'count_mismatches': None}
    counts: Counter[str] = Counter()
    boards: Counter[str] = Counter()
    missing_traces = mismatches = 0
    events = reported = 0
    lower_horizon_boards = 0
    for row in rows:
        trace, calls = row.get('trace'), integer(row.get('candidate_generation_calls'))
        if not isinstance(trace, list) or calls is None:
            missing_traces += 1
            continue
        generation = [event for event in trace if isinstance(event, dict) and event.get('event') == 'generation']
        events += len(generation)
        reported += calls
        mismatches += len(generation) != calls
        seen = set()
        for event in generation:
            k = integer(event.get('k'))
            name = str(k) if k is not None else 'unknown'
            counts[name] += 1
            seen.add(name)
        boards.update(seen)
        lower_horizon_boards += bool(seen & {'1', '4'})
    return {'calls_by_k': dict(sorted(counts.items())), 'boards_using_k': dict(sorted(boards.items())),
            'boards_using_k1_or_k4': lower_horizon_boards, 'trace_events': events, 'reported_calls': reported,
            'missing_traces': missing_traces, 'count_mismatches': mismatches}


def summarize_cases(cases: list[Record], freeze: Record) -> list[Record]:
    """Summarise both family totals and source-specific totals."""
    result = []
    strata = sorted({(row['family'], row['source_family'], row['method']) for row in cases})
    family_strata = sorted({(row['family'], 'all', row['method']) for row in cases})
    for family, source, method in family_strata + strata:
        selected = [row for row in cases if row['family'] == family and row['method'] == method
                    and (source == 'all' or row['source_family'] == source)]
        complete = [row for row in selected if known(row)]
        jobs = [row['result'] for row in complete]
        successes = [case['result'] for case in complete if joint_success(case, freeze)]
        late = [case['result'] for case in complete if case['result'].get('solved') is True
                and not joint_success(case, freeze)]
        failures = Counter(str(row.get('failure_reason') or 'unspecified') for row in jobs
                           if not row.get('solved'))
        lengths = [integer(row.get('solution_length')) for row in successes]
        walls = [numeric(row.get('wall_seconds')) for row in jobs]
        statuses = Counter((row['result'] or {}).get('status', 'missing') for row in selected)
        result.append({'family': family, 'source_family': source, 'method': method,
                       'planned': len(selected), 'known': len(complete), 'solved': len(successes),
                       'solutions_found_after_joint_budget': len(late),
                       'completed': statuses['completed'], 'timeouts': statuses['timeout'],
                       'infra_errors': statuses['infra_error'], 'missing': statuses['missing'],
                       'solved_per_planned_lower_bound': len(successes) / len(selected) if selected else None,
                       'solved_per_known': len(successes) / len(complete) if complete else None,
                       'median_solution_length': statistics.median(lengths) if lengths else None,
                       'mean_wall_seconds': statistics.mean(walls) if walls and all(x is not None for x in walls) else None,
                       'total_transitions': known_sum(jobs, 'transitions'),
                       'total_policy_calls': known_sum(jobs, 'policy_calls'),
                       'total_policy_input_tokens': input_token_sum(
                           jobs, 'policy_input_tokens', 'policy_calls', 146 if method == 'direct_policy' else 291),
                       'total_generation_calls': known_sum(jobs, 'candidate_generation_calls'),
                       'total_candidate_input_tokens': input_token_sum(
                           jobs, 'candidate_input_tokens', 'candidate_generation_calls', 145),
                       'total_candidate_tokens': known_sum(jobs, 'candidate_tokens'),
                       'total_value_calls': known_sum(jobs, 'value_calls'),
                       'total_value_input_tokens': input_token_sum(jobs, 'value_input_tokens', 'value_calls', 146),
                       'total_expansions': known_sum(jobs, 'high_level_expansions'),
                       'candidates_proposed': known_sum(jobs, 'candidates_proposed'),
                       'candidates_assessed': known_sum(jobs, 'candidates_assessed'),
                       'candidates_valid': known_sum(jobs, 'candidates_valid'),
                       'controller_attempts': known_sum(jobs, 'controller_attempts'),
                       'controller_attempts_completed': known_sum(jobs, 'controller_attempts_completed'),
                       'controller_attempts_missed': known_sum(jobs, 'controller_attempts_missed'),
                       'controller_cutoffs': known_sum(jobs, 'controller_cutoffs'),
                       'controller_task_solved_before_hit': known_sum(jobs, 'controller_task_solved_before_hit'),
                       'controller_hits': known_sum(jobs, 'controller_hits'),
                       'noops': known_sum(jobs, 'noop_actions'),
                       'generation_seconds': known_sum(jobs, 'generation_seconds'),
                       'controller_seconds': known_sum(jobs, 'controller_seconds'),
                       'value_seconds': known_sum(jobs, 'value_seconds'), 'failure_reasons': dict(failures),
                       'candidate_outcomes': known_mapping_total(jobs, 'candidate_outcomes'),
                       'generation_horizons': generation_horizons(jobs),
                       'infrastructure_reasons': dict(Counter(
                           str(row['result'].get('failure_reason') or 'unspecified') for row in selected
                           if row['result'] is not None and row['result'].get('status') == 'infra_error'))})
    return result


def success_curves(cases: list[Record], freeze: Record) -> list[Record]:
    """Success at each observed event; no interpolation or smoothing."""
    rows = []
    for family in FAMILIES:
        for method in methods_from_freeze(freeze):
            selected = [case for case in cases if case['family'] == family and case['method'] == method]
            if not selected:
                continue
            for axis, limit_field in (('transitions', 'transition_budget'), ('wall_seconds', 'per_job_wall_seconds')):
                limit = numeric(freeze.get(limit_field))
                if limit is None or limit <= 0:
                    continue
                events = {0.0, limit}
                for case in selected:
                    result = case['result']
                    if not joint_success(case, freeze):
                        continue
                    value = numeric(result.get('solve_at_transition' if axis == 'transitions' else 'solve_wall_seconds'))
                    if axis == 'wall_seconds' and value is None:
                        value = numeric(result.get('wall_seconds'))
                    if value is not None and 0 <= value <= limit:
                        events.add(value)
                for threshold in sorted(events):
                    successes = sum(at_budget(row, axis, threshold, freeze) for row in selected)
                    rows.append({'family': family, 'method': method, 'axis': axis, 'budget': threshold,
                                 'planned_boards': len(selected), 'known_boards': sum(known(row) for row in selected),
                                 'solved': successes, 'fraction_of_planned_lower_bound': successes / len(selected)})
    return rows


def paired_comparisons(cases: list[Record], freeze: Record, seed: int = 20261115) -> list[Record]:
    """Paired bootstrap at the final common joint budget, clustered by layout."""
    rows = []
    for family in FAMILIES:
        selected = [case for case in cases if case['family'] == family]
        by_key = {(case['layout_id'], case['method']): case for case in selected}
        layouts = sorted({case['layout_id'] for case in selected})
        methods = {case['method'] for case in selected}
        for left, right in (('adasubs', 'ksubs'), ('adasubs', 'direct_policy'), ('ksubs', 'direct_policy')):
            if left not in methods or right not in methods:
                continue
            differences = [float(joint_success(by_key[(layout, right)], freeze))
                           - float(joint_success(by_key[(layout, left)], freeze)) for layout in layouts
                           if (layout, left) in by_key and (layout, right) in by_key
                           and known(by_key[(layout, left)]) and known(by_key[(layout, right)])]
            generator = random.Random(f'{seed}:{family}:{left}:{right}')
            interval = bootstrap(differences, generator, count=2000) if differences else None
            rows.append({'family': family, 'left': left, 'right': right, 'planned_pairs': len(layouts),
                         'known_pairs': len(differences),
                         'difference_right_minus_left': statistics.mean(differences) if differences else None,
                         'ci_low': interval[0] if interval else None, 'ci_high': interval[1] if interval else None})
    return rows


def trace_comparisons(cases: list[Record], freeze: Record) -> list[Record]:
    """Compare saved AdaSubS/kSubS outcomes and traces, omitting only timestamps."""
    if not {'adasubs', 'ksubs'} <= set(methods_from_freeze(freeze)):
        return []
    def untimed(trace: Any) -> Any:
        if not isinstance(trace, list):
            return None
        return [{key: value for key, value in event.items() if not key.endswith('_seconds') and key != 'seconds'}
                if isinstance(event, dict) else event for event in trace]
    rows = []
    for family in FAMILIES:
        family_cases = [case for case in cases if case['family'] == family]
        keys = {(case['layout_id'], case['method']): case for case in family_cases}
        layouts = sorted({case['layout_id'] for case in family_cases})
        pairs = [(keys[(layout, 'adasubs')], keys[(layout, 'ksubs')]) for layout in layouts
                 if (layout, 'adasubs') in keys and (layout, 'ksubs') in keys
                 and known(keys[(layout, 'adasubs')]) and known(keys[(layout, 'ksubs')])]
        traces = [(untimed(a['result'].get('trace')), untimed(b['result'].get('trace'))) for a, b in pairs]
        a_only = [(a, b) for a, b in pairs if joint_success(a, freeze) and not joint_success(b, freeze)]
        def used_shorter_horizon(case: Record) -> bool:
            trace = case['result'].get('trace')
            return isinstance(trace, list) and any(isinstance(event, dict) and event.get('event') == 'generation'
                                                   and event.get('k') in (1, 4) for event in trace)
        rows.append({'family': family, 'planned_pairs': len(layouts), 'known_pairs': len(pairs),
                     'same_solved': sum(joint_success(a, freeze) == joint_success(b, freeze) for a, b in pairs),
                     'same_raw_solved': sum(a['result']['solved'] == b['result']['solved'] for a, b in pairs),
                     'adasubs_only_success': len(a_only),
                     'ksubs_only_success': sum(not joint_success(a, freeze) and joint_success(b, freeze) for a, b in pairs),
                     'adasubs_only_used_k1_or_k4': sum(used_shorter_horizon(a) for a, _ in a_only),
                     'adasubs_only_used_shorter_and_ksubs_queue_exhausted': sum(
                         used_shorter_horizon(a) and b['result'].get('failure_reason') == 'queue_exhausted'
                         for a, b in a_only),
                     'adasubs_only_ksubs_timeout': sum(b['result'].get('status') == 'timeout' for _, b in a_only),
                     'same_status': sum(a['result']['status'] == b['result']['status'] for a, b in pairs),
                     'same_transition_count': sum(a['result']['transitions'] == b['result']['transitions'] for a, b in pairs),
                     'trace_pairs_available': sum(a is not None and b is not None for a, b in traces),
                     'same_trace_ignoring_times': sum(a is not None and b is not None and a == b for a, b in traces)})
    return rows


def curve_line_style(method: str, partial: bool) -> str:
    if partial:
        return ':'
    return '--' if method == 'ksubs' else '-'


def plot_curves(curves: list[Record], freeze: Record, output: Path) -> None:
    """Draw observed success curves against simulation work and wall time."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), layout='constrained', sharey=True)
    labels = {'transitions': 'Charged primitive transitions', 'wall_seconds': 'Per-board time (s; no model loading)'}
    for i, family in enumerate(FAMILIES):
        for j, axis in enumerate(('transitions', 'wall_seconds')):
            ax = axes[i, j]
            for method in methods_from_freeze(freeze):
                values = sorted((row for row in curves if row['family'] == family and row['method'] == method
                                 and row['axis'] == axis), key=lambda row: row['budget'])
                if not values or not values[0]['known_boards']:
                    continue
                missing = values[0]['known_boards'] < values[0]['planned_boards']
                label = METHOD_LABELS.get(method, method) + (' (partial)' if missing else '')
                ax.step([row['budget'] for row in values], [row['fraction_of_planned_lower_bound'] for row in values],
                        where='post', label=label, color=COLOURS.get(method), linewidth=1.5,
                        linestyle=curve_line_style(method, missing))
            planned = next((row['planned_boards'] for row in curves if row['family'] == family), 0)
            ax.set_title(f'{family} · {planned} planned boards')
            ax.set_xlabel(labels[axis])
            if j == 0:
                ax.set_ylabel('Solved within both caps / planned')
            ax.set_ylim(-.025, 1.025)
            ax.grid(alpha=.2)
    legend = {}
    for ax in axes.flat:
        handles, names = ax.get_legend_handles_labels()
        for handle, name in zip(handles, names, strict=True):
            legend.setdefault(name, handle)
    if legend:
        fig.legend(list(legend.values()), list(legend), loc='outside lower center',
                   ncol=min(len(legend), 4), frameon=False)
    for extension in ('pdf', 'svg', 'png'):
        path = output / f'stage_05_success.{extension}'
        temp = output / f'.stage_05_success.{os.getpid()}.{extension}.tmp'
        try:
            fig.savefig(temp, format=extension, dpi=180, bbox_inches='tight')
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    plt.close(fig)


def ratio(numerator: Any, denominator: Any) -> str:
    n, d = integer(numerator), integer(denominator)
    if n is None or d is None:
        return 'n/a'
    return f'{n}/{d}' + (f' ({100 * n / d:.1f}%)' if d else '')


def solved_cell(row: Record) -> str:
    """Do not make an entirely absent method look like zero observed successes."""
    if not row['known']:
        return f"n/a (0/{row['planned']} observed)"
    if row['known'] < row['planned']:
        return f"≥{ratio(row['solved'], row['planned'])}; {row['known']}/{row['planned']} observed"
    return ratio(row['solved'], row['planned'])


def shown(value: Any, digits: int = 2) -> str:
    value = numeric(value)
    return 'n/a' if value is None else f'{value:.{digits}f}'


def display_count(value: Any) -> str:
    value = numeric(value)
    if value is None:
        return 'n/a'
    return str(int(value)) if value.is_integer() else f'{value:.2f}'


def history_lines(history: Record) -> list[str]:
    first, second, third, fourth = (history[f'stage_0{i}'] for i in range(1, 5))
    lines = ['## Stages 1–4: what was measured', '',
             'All historical cohorts below are exploratory. The full-state CLLP is given a complete '
             'target board, whereas the Qwen generator proposes predicate intents; these are different '
             'input contracts and should not be pooled. “Fresh response” means a new model response, not '
             'a previously unseen board.', '',
             '### 1. Frozen full-state controllers', '',
             f"{first.get('completed_trials', 'n/a')}/{first.get('planned_trials', 'n/a')} planned trials "
             'were completed across five CLLP checkpoints and the separate direct policy. At L=32 '
             'primitive actions, the following are CLLP target-board hits, not complete-puzzle '
             'solutions; k is a demonstration action offset. Available denominators vary with '
             'trajectory length; a missing offset is not a failed attempt.', '',
             '| Family | Controller | k | Target hits / available |', '|---|---|---:|---:|']
    for row in sorted(first['selected'], key=lambda x: (x['family'], int(x['k']), x['model_id'])):
        lines.append(f"| {row['family']} | {row['model_id']} | {row['k']} | "
                     f"{row['hits']}/{row['complete']} |")
    lines += ['', '### 2. Controller-derived temporal profiles', '',
              f"The fixed full-state profiles contain {second.get('evaluation_attempts', 'n/a')} evaluation "
              f"attempts and {second.get('train_attempts', 'n/a')} training-role attempts. The following "
              'offset-versus-duration rank comparisons include only targets reached by *both* controllers. '
              'The mean includes all contexts with at least one shared success, including any singleton '
              'with zero difference. The number with at least two distinct durations is shown separately. '
              'They do not evaluate relative-scale conditioning or show that it improves planning.', '',
              '| Family | Controller | Contexts in mean | Contexts with ≥2 distinct durations | '
              'Mean absolute difference u(d)−u(k) | Strict inversions / comparable pairs |',
              '|---|---|---:|---:|---:|---:|']
    for row in sorted(second['conditional_rank'], key=lambda x: (x['family'], x['controller'])):
        try:
            mean = f"{float(row['mean_abs_u_difference']):.3f}"
        except (ValueError, TypeError):
            mean = 'n/a'
        lines.append(f"| {row['family']} | {row['controller']} | {row['error_contexts']} | "
                     f"{row['contexts_with_two_distinct_d']} | {mean} | {row['inversions']}/{row['strict_comparable']} |")
    fresh = third.get('fresh_generation_status') or {}
    overlap = third.get('overlap') or {}
    data = fourth.get('dataset') or {}
    narrow_layouts = (data.get('training_layouts') or {}).get('narrow')
    wide_layouts = (data.get('training_layouts') or {}).get('wide')
    narrow_examples = (data.get('training_examples') or {}).get('narrow')
    wide_examples = (data.get('training_examples') or {}).get('wide')
    common = data.get('common_original_examples')
    lines += ['', 'The earlier analysis also found that empirical scale changes when the candidate pool or '
              'action limit changes. M=1 is a deterministic observed hit/miss, not a precise stochastic '
              'success-rate estimate.', '', '### 3. Does Qwen use the board?', '',
              f"The fresh-generation audit completed {fresh.get('completed', 'n/a')}/"
              f"{fresh.get('planned', 'n/a')} responses on {overlap.get('fresh_boards', 'n/a')} "
              'original boards with transformed views. The boards had already appeared in the '
              'campaign. Permuting outputs and transforming boards probes association with the input; '
              'many different intents can be valid, and exact disagreement after a transform is not '
              'a proof of misunderstanding. BFS feasibility is not learned-controller success. '
              '[Full grounding audit](stage_03_report.md).', '']
    transformations = {row['transform']: row for row in third.get('x4_checkpoint_transforms') or []}
    if all(name in transformations for name in ('identity', 'rot90_ccw', 'horizontal_reflection')):
        def fraction(name: str) -> str:
            row = transformations[name]
            return f"{row['valid_nontrivial']}/{row['requested_outputs']}"
        lines += ['For the old validation-selected LoRA on the same ×4 layouts, valid outputs were '
                  f"{fraction('identity')} in the original orientation, {fraction('rot90_ccw')} after "
                  f"rotation and {fraction('horizontal_reflection')} after reflection. This is a paired "
                  'diagnostic, not a complete-task measurement.', '']
    icl = third.get('x10_icl_original') or []
    if icl:
        def total(field: str) -> int:
            return sum(int(row[field]) for row in icl)
        lines += ['In the fixed two-example ICL configuration on the mixed ×10 sources, '
                  f"{total('valid_nontrivial')}/{total('requested_outputs')} proposed intents were valid, "
                  f"{total('whole_goal_requests')}/{total('requested_outputs')} requested the whole puzzle goal, "
                  f"and bounded BFS found {total('bounded_paths_found')}/{total('requested_outputs')} paths. "
                  'This illustrates why validity alone does not measure useful planning.', '']
    lines += ['![Stage 3 board-conditioning diagnostics](stage_03_figure.png)', '',
              '### 4. Layout diversity and coordinate augmentation', '',
              f"One seed per new adapter; NARROW has {narrow_examples if narrow_examples is not None else 'unknown'} "
              f"examples from {narrow_layouts if narrow_layouts is not None else 'unknown'} layouts, WIDE "
              f"and WIDE+D4 use the same {wide_examples if wide_examples is not None else 'unknown'} examples "
              f"from {wide_layouts if wide_layouts is not None else 'unknown'} layouts. All adaptation data are ×4. "
              f"NARROW and WIDE share {common if common is not None else 'an unknown number of'} example IDs; "
              'the comparison changes examples as well as layout count. The evaluation boards were '
              'previously inspected. The old adapter is a historical reference.', '',
              '| Family | Generator | Valid intent / requested | Bounded BFS path / requested |',
              '|---|---|---:|---:|']
    for row in fourth.get('identity_figure') or []:
        lines.append(f"| {row['family']} | {row['method']} | "
                     f"{ratio(row.get('valid_outputs'), row.get('planned_outputs'))} | "
                     f"{ratio(row.get('bounded_paths_found'), row.get('planned_outputs'))} |")
    coverage = [row for row in fourth.get('predicate_coverage') or [] if row.get('method') == 'wide_aug']
    if len(coverage) == 2 and all(row.get('parseable_outputs') and
                                  (row.get('predicate_presence_parseable') or {}).get('box_at') == row['parseable_outputs']
                                  for row in coverage):
        lines += ['', 'Every JSON-parseable WIDE+D4 response in the original orientations still contained '
                  '`box_at`; many also added a second predicate. Invalid or already satisfied '
                  'predicates remain in the denominator.']
    paired = [row for row in fourth.get('paired_family') or [] if row.get('first') == 'wide'
              and row.get('second') == 'wide_aug' and row.get('metric') == 'valid' and row.get('transform') == 'identity']
    if len(paired) == 2 and all(row.get('ci_low') is not None and row.get('ci_high') is not None
                                and row['ci_low'] <= 0 <= row['ci_high'] for row in paired):
        lines += ['The WIDE versus WIDE+D4 validity differences have paired 95% board-bootstrap '
                  'intervals spanning zero in both families.']
    lines += ['',
              '![Stage 4 repeat-coordinate diagnostics](stage_04_repetition.png)', '',
              '[Stage 1 full report](stage_01_report.md) · [Stage 2 full report](stage_02_report.md) · '
              '[Stage 3 full report](stage_03_report.md) · [Stage 4 full report](stage_04_report.md).', '']
    return lines


def recorded_commands(supervisor: Record | None) -> list[Record]:
    commands = []
    for stage in (supervisor or {}).get('stages', []):
        argv = stage.get('command')
        if not isinstance(argv, list) or not argv:
            continue
        # The campaign is offline. Do not leak a real credential should an
        # unrelated command be registered accidentally.
        safe = []
        for arg in argv:
            arg = str(arg)
            name, separator, value = arg.partition('=')
            if separator and (name.endswith('_KEY') or name.endswith('_TOKEN')) and value != 'offline-test-key':
                arg = name + '=<redacted>'
            safe.append(arg)
        commands.append({'stage': stage.get('name'), 'status_at_report_snapshot': stage.get('status'),
                         'started_utc': stage.get('started_utc'), 'finished_utc': stage.get('finished_utc'),
                         'process_wall_seconds': elapsed_utc(stage.get('started_utc'), stage.get('finished_utc')),
                         'command': shlex.join(safe)})
    return commands


def elapsed_utc(start: str | None, end: str | None) -> float | None:
    """Return the difference in logged UTC timestamps; the clock may be adjusted."""
    if not start or not end:
        return None
    try:
        first = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
        last = datetime.datetime.fromisoformat(end.replace('Z', '+00:00'))
        result = (last - first).total_seconds()
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


def evaluation_process_statuses(campaign: Path) -> list[Record]:
    """Read runner-monotonic total times, including each runner's model loading."""
    rows = []
    for method, directory in (('adasubs', 'adasubs_evaluation'), ('ksubs', 'ksubs_evaluation'),
                              ('direct_policy', 'policy_evaluation'), ('primitive_astar', 'primitive_evaluation')):
        path = campaign / 'stage_05' / directory / 'status.json'
        status = optional_json(path)
        if status is None:
            continue
        rows.append({'method': method, 'state': status.get('state'), 'planned': status.get('planned'),
                     'completed': status.get('completed'), 'elapsed_seconds': status.get('elapsed_seconds'),
                     'peak_host_memory_percent': status.get('peak_host_memory_percent'),
                     'peak_process_tree_rss_bytes': status.get('peak_process_tree_rss_bytes'),
                     'peak_swap_used_bytes': status.get('peak_swap_used_bytes'), 'path': str(path)})
    return rows


def replay_audit_summary(campaign: Path, freeze: Record) -> Record:
    """Verify completed offline audit summaries without adding their costs to methods."""
    stage = campaign / 'stage_05'
    path = stage / 'final_audit_manifest.json'
    manifest = optional_json(path)
    interruption_path = stage / 'final_audit_baselines_interruption.json'
    interruption = optional_json(interruption_path)
    other_audits = []
    for name in ('early_trace_sanity.json', 'audit_dev_smoke.json', 'audit_dev_updated.json',
                 'audit_primitive_smoke.json', 'audit_fast_preflight.json'):
        audit_path = stage / name
        value = optional_json(audit_path)
        if value is None:
            continue
        checks = value.get('checks') if isinstance(value.get('checks'), list) else None
        other_audits.append({'file': str(audit_path), 'sha256': sha256(audit_path),
                             'state': value.get('state', 'checks recorded'),
                             'present_jobs': value.get('present_jobs', len(checks) if checks is not None else None),
                             'expected_jobs': value.get('expected_jobs', len(checks) if checks is not None else None),
                             'offline_audit_transitions': value.get('offline_audit_transitions',
                                                                   value.get('additional_audit_simulator_transitions'))})
    issues: list[str] = []
    if manifest is not None:
        if manifest.get('board_sha256') != freeze.get('input_boards_sha256'):
            issues.append('Combined audit board hash differs from the frozen board file')
        methods = manifest.get('methods')
        if not isinstance(methods, dict):
            methods = {}
            issues.append('Combined audit has no valid methods mapping')
        for name, row in methods.items():
            if not isinstance(row, dict):
                issues.append(f'Invalid audit metadata for {name}')
                continue
            audit_path = Path(row.get('audit_file') or '')
            expected = row.get('audit_sha256')
            if not audit_path.is_file() or not isinstance(expected, str) or sha256(audit_path) != expected:
                issues.append(f'Cannot verify audit file for {name}')
        if set(manifest.get('expected_methods') or []) != set(freeze.get('methods') or []):
            issues.append('Combined audit methods differ from the frozen evaluation methods')
        if manifest.get('state') == 'pass' and set(methods) != set(freeze.get('methods') or []):
            issues.append('Combined audit reports pass while some frozen methods are absent')
        if manifest.get('state') == 'pass' and manifest.get('issues'):
            issues.append('Combined audit reports pass alongside unresolved issues')
        replayed = [integer(row.get(field)) for row in methods.values() if isinstance(row, dict)
                    for field in ('recorded_transitions_replayed', 'solution_transitions_replayed')]
        if replayed and all(value is not None for value in replayed):
            if sum(replayed) != integer(manifest.get('offline_audit_transitions')):
                issues.append('Combined offline transition total differs from per-method audit counts')
        prior = manifest.get('interrupted_prior_audit')
        if isinstance(prior, dict):
            prior_path = Path(prior.get('path') or '')
            if not prior_path.is_file() or sha256(prior_path) != prior.get('sha256'):
                issues.append('Cannot verify the interrupted-audit record')
        return {'state': manifest.get('state') if not issues else 'partial', 'manifest_path': str(path),
                'methods': methods, 'offline_audit_transitions': manifest.get('offline_audit_transitions'),
                'interrupted_prior_audit': prior, 'interruption_detail': interruption,
                'other_audits': other_audits, 'issues': [*manifest.get('issues', []), *issues]}
    baseline_path = stage / 'final_audit_baselines.json'
    baseline = optional_json(baseline_path)
    methods = {}
    if baseline:
        if baseline.get('boards_sha256') != freeze.get('input_boards_sha256'):
            issues.append('Baseline audit board hash differs from the frozen board file')
        if baseline.get('state') != 'pass' or baseline.get('errors') or baseline.get('missing_jobs'):
            issues.append('Baseline audit is incomplete or reports errors')
        methods = {name: {'planned_boards': baseline.get('frozen_boards'), **row,
                          'audit_file': str(baseline_path), 'audit_sha256': sha256(baseline_path)}
                   for name, row in baseline.get('by_method', {}).items()}
    prior = None
    if interruption:
        prior = {'path': str(interruption_path), 'sha256': sha256(interruption_path),
                 'offline_transitions': interruption.get('partial_offline_transition_count'),
                 'note': interruption.get('partial_offline_transition_count_reason')}
    return {'state': 'partial' if baseline or interruption else 'not_available', 'manifest_path': None,
            'methods': methods, 'offline_audit_transitions': baseline.get('offline_audit_transitions') if baseline else None,
            'interrupted_prior_audit': prior, 'interruption_detail': interruption,
            'other_audits': other_audits, 'issues': issues}


def prior_stages_audit(campaign: Path) -> Record:
    """Verify the independent audit and each source it names."""
    path = campaign / 'stage_05/prior_stages_independent_audit.json'
    audit = optional_json(path)
    if audit is None:
        return {'state': 'not_available', 'path': str(path), 'sha256': None, 'sources': 0, 'issues': []}
    issues = []
    script = campaign / 'stage_05/prior_stages_independent_audit.py'
    if not script.is_file() or sha256(script) != audit.get('script_sha256'):
        issues.append('Prior-stage audit script hash differs')
    sources = audit.get('source_sha256') or {}
    for name, expected in sources.items():
        source = campaign / name
        if not source.is_file() or sha256(source) != expected:
            issues.append(f'Prior-stage audit source hash differs: {name}')
    state = audit.get('status') if not issues else 'partial'
    return {'state': state, 'path': str(path), 'sha256': sha256(path), 'sources': len(sources), 'issues': issues}


def evaluation_contract_audit(campaign: Path, freeze_path: Path, freeze: Record) -> Record:
    """Verify the independent comparison of execution contracts with the freeze."""
    path = campaign / 'stage_05/evaluation_contract_audit.json'
    audit = optional_json(path)
    if audit is None:
        return {'state': 'not_available', 'path': str(path), 'sha256': None, 'contracts': {}, 'issues': []}
    issues = []
    if audit.get('boards_sha256') != freeze.get('input_boards_sha256'):
        issues.append('Contract audit board hash differs from the freeze')
    if not freeze_path.is_file() or audit.get('frozen_evaluation_sha256') != sha256(freeze_path):
        issues.append('Contract audit freeze hash differs')
    contracts = audit.get('contracts') or {}
    if not isinstance(contracts, dict):
        contracts = {}
        issues.append('Contract audit has no valid contracts mapping')
    if len(contracts) != len(freeze.get('methods') or []):
        issues.append('Contract audit does not cover every frozen method')
    for name, row in contracts.items():
        if not isinstance(row, dict):
            issues.append(f'Invalid execution contract metadata: {name}')
            continue
        contract = Path(row.get('path') or '')
        if not contract.is_file() or sha256(contract) != row.get('sha256'):
            issues.append(f'Execution contract hash differs: {name}')
    return {'state': audit.get('state') if not issues else 'partial', 'path': str(path),
            'sha256': sha256(path), 'contracts': contracts, 'issues': issues}


def repository_snapshot() -> Record:
    """Record the local checkout state, without modifying it or contacting a remote."""
    root = Path(__file__).resolve().parents[1]
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True,
                                  check=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git('status', '--short', '--untracked-files=all')
    return {'branch': git('branch', '--show-current'), 'head': git('rev-parse', 'HEAD'),
            'head_stat': git('show', '--stat', '--oneline', '--format=%h %s', 'HEAD'),
            'changed_files': status.splitlines() if status else [], 'status_available': status is not None}


def lifecycle_snapshot(manifest: Record) -> Record:
    stage = (manifest.get('stages') or {}).get('05') or {}
    return {key: stage.get(key) for key in ('status', 'phase', 'owned_processes_stopped', 'supervisor_status', 'closed_utc')}


def stage_five_lines(freeze: Record, summary: list[Record], pairs: list[Record], traces: list[Record], campaign: Path,
                     commands: list[Record], process_statuses: list[Record], merge: Record,
                     board_manifest: Record | None, inventory: list[Record], qa: Record, contract_audit: Record,
                     boards_link: str = 'stage_05/boards.jsonl', plan_error: str | None = None,
                     invalid_records: list[str] | None = None) -> list[str]:
    lines = ['# Stage 5 — complete-puzzle full-state baselines', '',
             'These are full-state Sokoban baselines, not predicate-intent controllers and not a test '
             'of policy-relative u. The direct policy receives only the current board; CLLP receives '
             'the current board and a complete generated target board. The old horizon selection '
             'must not be relabelled as the proposal’s requested relative scale.', '',
             f"Frozen simulator-transition budget: `{freeze.get('transition_budget', 'not frozen')}`; "
             f"per-job wall limit: `{freeze.get('per_job_wall_seconds', 'not frozen')}` seconds; "
             f"fixed-k horizon chosen on development data: `{freeze.get('ksubs_horizon', 'not recorded')}`; "
             f"expansion safety limit: `{freeze.get('max_expansions', 'not recorded')}`. "
             'The transition budget covers attempted primitive actions, including failed actions and '
             'no-ops, as well as any simulator work charged by the runner. Neural calls and elapsed '
             'time are listed separately; equal transition budgets do not imply equal compute.', '',
             'Every frozen board/method pair is retained. `timeout` means the declared per-job wall '
             'limit was reached. Missing shards and infrastructure errors are unknown outcomes, '
             'not failures; the solved/planned figure is a lower bound when any are missing. '
             'Fixed-k kSubS uses a dashed line; a dotted line and a partial label mark an incomplete '
             'method. A method with no known outcomes has no line. Overlapping curves show equal '
             'cumulative success fractions along that interval, not a missing method. '
             'A timeout may censor a transition-only curve before its nominal action limit.', '',
             f"Merged shard manifest: `{merge['state']}`; hash verification: `{merge['hash_verified']}`; "
             f"rejected shard references: {len(merge['rejected'])}; unlisted files: {len(merge['unlisted'])}; "
             f"unavailable source copies: {len(merge['source_unavailable'])}. "
             'Unverified or unlisted shards are not treated as behavioural failures.', '',
             '![Task success versus charged transitions and time](stage_05/stage_05_success.png)', '',
             'The curves use actual recorded solution thresholds, with no interpolation; per-board '
             'wall time excludes one-time worker model loading and is independent of transition count. '
             'Each success must satisfy both the transition and wall caps, including when plotting '
             'only one of those axes. A plan returned after the wall cap is counted separately as late, '
             'not as an infrastructure error or a within-budget success. '
             'Total supervised process times, including model loading, are listed below. Source distributions differ between '
             '×4 and ×10, so the family gap is not a causal effect of box count. When a runner '
             'records only final wall time, it is used as the observed time to return a solution.', '',
             '| Family | Source | Method | Joint-budget solved / planned | Late solutions | Completed | Timeouts | '
             'Infrastructure errors | Missing | Median solution length | Mean wall seconds |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    if 'primitive_astar' in methods_from_freeze(freeze):
        lines[2:2] = ['Primitive A* is a secondary baseline using the shared value heuristic; '
                      'it does not change the three primary comparisons.', '']
    convention = freeze.get('value_convention')
    if convention:
        lines[2:2] = [f'Value ranking: {convention}.', '']
    controller_limit = freeze.get('controller_limit')
    if controller_limit:
        lines[2:2] = [f'Fixed controller attempt limit: {controller_limit}.', '']
    if freeze.get('generation_simulator_transitions') is not None:
        lines[2:2] = [f"Generator-side simulator transitions per proposal batch: "
                      f"{freeze['generation_simulator_transitions']} (as recorded in the freeze).", '']
    if board_manifest:
        lines[2:2] = [f"Board selection: {board_manifest.get('selection', 'not recorded')} "
                      f"Initially solved starts: {board_manifest.get('initially_solved', 'unknown')}. "
                      f"Source: `{board_manifest.get('source', 'unknown')}`. "
                      f"Cohort: {board_manifest.get('test_cohort', 'exposure not recorded')}. "
                      'Future states, demonstration actions and offsets are not planner inputs.', '']
    checkpoints = freeze.get('checkpoints') or {}
    if checkpoints:
        provenance = ['', '## Frozen model checkpoints', '',
                      '| Component | Local checkpoint | Weights SHA256 | Configuration SHA256 |',
                      '|---|---|---|---|']
        for name, spec in sorted(checkpoints.items()):
            provenance.append(f"| {name} | `{spec.get('path', 'unknown')}` | "
                              f"`{spec.get('weights_sha256', 'unknown')}` | `{spec.get('config_sha256', 'unknown')}` |")
        provenance += ['', f"Frozen decoding: `{freeze.get('decoding', {})}`. Source and checkpoint hashes "
                       'are recorded in the linked freeze.', '']
        marker = '| Family | Source | Method | Joint-budget solved / planned | Late solutions | Completed | Timeouts | Infrastructure errors | Missing | Median solution length | Mean wall seconds |'
        lines[lines.index(marker):lines.index(marker)] = provenance
    for row in summary:
        lines.append(f"| {row['family']} | {row['source_family']} | {row['method']} | "
                     f"{solved_cell(row)} | {row['solutions_found_after_joint_budget']} | "
                     f"{row['completed']} | {row['timeouts']} | "
                     f"{row['infra_errors']} | {row['missing']} | {shown(row['median_solution_length'], 1)} | "
                     f"{shown(row['mean_wall_seconds'])} |")
    if inventory:
        lines += ['', '## Initial puzzle states', '',
                  'These are complete trajectory starting boards selected by the frozen rank rule, '
                  'not fragments selected for being close to a solution. The distributions describe '
                  'the sources and were not used to choose boards by model outcome.', '',
                  '| Family | Source | Boards | Initial states checked | Boxes already on goals: board counts |',
                  '|---|---|---:|---:|---|']
        for row in inventory:
            lines.append(f"| {row['family']} | {row['source_family']} | {row['planned_boards']} | "
                         f"{row['checked_initial_states']} | `{row['initial_boxes_on_goals_distribution']}` |")
    if merge['rejected'] or merge['unlisted'] or merge['source_unavailable']:
        lines += ['', f"Shard verification issues: `{merge['rejected']}`; unlisted files: `{merge['unlisted']}`; "
                  f"unavailable source copies: `{merge['source_unavailable']}`.", '']
    if plan_error:
        lines += ['', f'The frozen board plan could not be validated: `{plan_error}`. '
                  'Stage-five outcome rates are unavailable, not zero.', '']
    if invalid_records:
        lines += ['', 'The following malformed or conflicting shard records were excluded; their '
                  'board/method outcomes remain unknown:', '', *[f'- `{item}`' for item in invalid_records], '']
    lines += ['', 'Paired comparisons below use only boards with observed outcomes for both methods '
              'and bootstrap original canonical layouts. A degenerate empirical interval from all '
              'successes or all failures does not imply certainty about the population.', '',
              '| Family | Comparison | Known / planned pairs | Difference | 95% board bootstrap |',
              '|---|---|---:|---:|---:|']
    for row in pairs:
        difference = row['difference_right_minus_left']
        interval = (f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
                    if row['ci_low'] is not None and row['ci_high'] is not None else 'n/a')
        shown_difference = f'{difference:+.3f}' if difference is not None else 'n/a'
        lines.append(f"| {row['family']} | {row['right']} − {row['left']} | "
                     f"{row['known_pairs']}/{row['planned_pairs']} | {shown_difference} | {interval} |")
    lines += ['', '## Costs and failure categories', '',
              'Totals below aggregate known completed or timed-out jobs. `n/a` means a required '
              'measurement is absent, not zero. Candidate validity and controller hit rates have '
              'different denominators; invalid proposals are not controller misses. Search-controller '
              'input tokens are 291 per CLLP call and value input tokens 146 per call under the '
              'verified 12×12 tokenizer; generator inputs are 145 per call. Generated token totals '
              'include special tokens and padding in returned sequences. Controller hits are divided '
              'by completed attempts, excluding cutoff attempts and task solutions reached before '
              'the proposed target. Started attempts and cutoffs are reported separately.', '',
              '| Family | Source | Method | Transitions | Policy calls / input tokens | '
              'Generation calls / input / output tokens | Value calls / input tokens | Expansions | Proposed | '
              'Valid / assessed | Hits / completed | Started / cutoffs / task-solved-first | No-ops |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['family']} | {row['source_family']} | {row['method']} | "
                     f"{display_count(row['total_transitions'])} | {display_count(row['total_policy_calls'])} / "
                     f"{display_count(row['total_policy_input_tokens'])} | {display_count(row['total_generation_calls'])} / "
                     f"{display_count(row['total_candidate_input_tokens'])} / {display_count(row['total_candidate_tokens'])} | "
                     f"{display_count(row['total_value_calls'])} / {display_count(row['total_value_input_tokens'])} | "
                     f"{display_count(row['total_expansions'])} | {display_count(row['candidates_proposed'])} | "
                     f"{ratio(row['candidates_valid'], row['candidates_assessed'])} | "
                     f"{ratio(row['controller_hits'], row['controller_attempts_completed'])} | "
                     f"{display_count(row['controller_attempts'])} / {display_count(row['controller_cutoffs'])} / "
                     f"{display_count(row['controller_task_solved_before_hit'])} | "
                     f"{display_count(row['noops'])} |")
    lines += ['', '| Family | Source | Method | Generation seconds | Controller seconds | Value seconds |',
              '|---|---|---|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['family']} | {row['source_family']} | {row['method']} | "
                     f"{shown(row['generation_seconds'])} | {shown(row['controller_seconds'])} | "
                     f"{shown(row['value_seconds'])} |")
    lines += ['', 'Generator usage is counted from actual `generation` trace events, not from '
              'the list of loaded checkpoints. These are the existing absolute horizons k, not '
              'policy-relative scale labels. A mismatch or missing trace is reported explicitly.', '',
              '| Family | Source | Method | Calls by k | Boards using k | Boards using k1 or k4 | '
              'Trace events / reported calls | Missing traces | Count mismatches |',
              '|---|---|---|---|---|---:|---:|---:|---:|']
    for row in summary:
        horizons = row['generation_horizons']
        lines.append(f"| {row['family']} | {row['source_family']} | {row['method']} | "
                     f"`{horizons['calls_by_k'] if horizons['calls_by_k'] is not None else 'n/a'}` | "
                     f"`{horizons['boards_using_k'] if horizons['boards_using_k'] is not None else 'n/a'}` | "
                     f"{display_count(horizons['boards_using_k1_or_k4'])} | "
                     f"{display_count(horizons['trace_events'])}/{display_count(horizons['reported_calls'])} | "
                     f"{display_count(horizons['missing_traces'])} | {display_count(horizons['count_mismatches'])} |")
    if traces:
        lines += ['', 'On paired boards, the following compares AdaSubS with fixed-k kSubS. '
                  'Trace equality removes timestamp and measured-duration fields only; it does not '
                  'assert that the methods always agree or that their elapsed times match.', '',
                  '| Family | Known / planned | Same joint-budget result | Same reported solution | '
                  'Same status | Same transition count | Equal untimed traces / available |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for row in traces:
            lines.append(f"| {row['family']} | {row['known_pairs']}/{row['planned_pairs']} | "
                         f"{row['same_solved']} | {row['same_raw_solved']} | "
                         f"{row['same_status']} | {row['same_transition_count']} | "
                         f"{row['same_trace_ignoring_times']}/{row['trace_pairs_available']} |")
        lines += ['', 'For boards solved within both caps by AdaSubS but not kSubS, the following '
                  'describes observed k use and stopping reasons. These overlaps suggest possible '
                  'mechanisms but do not establish causality.', '',
                  '| Family | AdaSubS-only / kSubS-only | Ada-only using k1 or k4 | '
                  'Ada-only using shorter k and kSubS queue exhausted | Ada-only with kSubS timeout |',
                  '|---|---:|---:|---:|---:|']
        for row in traces:
            lines.append(f"| {row['family']} | {row['adasubs_only_success']} / {row['ksubs_only_success']} | "
                         f"{row['adasubs_only_used_k1_or_k4']} | "
                         f"{row['adasubs_only_used_shorter_and_ksubs_queue_exhausted']} | "
                         f"{row['adasubs_only_ksubs_timeout']} |")
    lines += ['', 'Recorded unsolved reasons:', '']
    for row in summary:
        if row['source_family'] == 'all':
            lines.append(f"- {row['family']} / {row['method']}: job reasons `{row['failure_reasons']}`; "
                         f"infrastructure reasons `{row['infrastructure_reasons']}`; "
                         f"candidate outcomes `{row['candidate_outcomes'] if row['candidate_outcomes'] is not None else 'n/a'}`.")
    lines += ['', '## Offline replay and audit work', '',
              f"Independent execution-contract audit: `{contract_audit['state']}`, "
              f"{len(contract_audit['contracts'])} method contracts, SHA256 "
              f"`{contract_audit['sha256'] or 'unavailable'}`. "
              '[Contract audit](stage_05/evaluation_contract_audit.json).', '',
              f"Combined final audit state: `{qa['state']}`. Its completed audits account for "
              f"{display_count(qa['offline_audit_transitions'])} extra offline simulator transitions, "
              'which are separate from the methods’ per-board transition budgets and solution curves.', '',
              '| Method | Audited / planned boards | Recorded transitions replayed | '
              'Solution transitions replayed | Solutions verified |', '|---|---:|---:|---:|---:|']
    for name, row in sorted(qa['methods'].items()):
        if isinstance(row, dict):
            lines.append(f"| {name} | {display_count(row.get('jobs_audited'))}/{display_count(row.get('planned_boards'))} | "
                         f"{display_count(row.get('recorded_transitions_replayed'))} | "
                         f"{display_count(row.get('solution_transitions_replayed'))} | "
                         f"{display_count(row.get('solutions_verified'))} |")
    prior = qa.get('interrupted_prior_audit')
    if isinstance(prior, dict):
        count = prior.get('offline_transitions')
        count_text = 'unknown (not zero)' if count is None else display_count(count)
        detail = qa.get('interruption_detail') or {}
        reason = str(detail.get('reason', prior.get('note', 'not recorded'))).rstrip('.')
        lines += ['', 'An earlier offline baseline audit was interrupted because it reconstructed the '
                  f"environment unnecessarily on each transition. Its partial transition count is {count_text}. "
                  f"Recorded reason: {reason}. This was an audit process, not a solver failure.", '']
    if qa['issues']:
        lines += ['', f"Audit verification issues: `{qa['issues']}`.", '']
    if contract_audit['issues']:
        lines += ['', f"Execution-contract audit verification issues: `{contract_audit['issues']}`.", '']
    if qa['other_audits']:
        lines += ['', 'Other recorded offline checks (development checks can replay the same trace and '
                  'are separate computations, not independent evaluation cases):', '',
                  '| Audit | State | Present / expected jobs | Extra offline transitions |', '|---|---|---:|---:|']
        for row in qa['other_audits']:
            lines.append(f"| `{Path(row['file']).name}` | {row['state']} | "
                         f"{display_count(row['present_jobs'])}/{display_count(row['expected_jobs'])} | "
                         f"{display_count(row['offline_audit_transitions'])} |")
    lines += ['', 'These descriptive counts can suggest whether target validity, controller hits or '
              'search dominates observed cost; they do not establish a causal bottleneck. Missing '
              'methods and incomplete jobs must not silently disappear.', '',
              '[Frozen evaluation configuration](stage_05/evaluation_freeze.json) · '
              f'[Frozen boards]({boards_link}) · [Summary CSV](stage_05/stage_05_summary.csv) · '
              '[Curves CSV](stage_05/stage_05_curves.csv) · [Paired CSV](stage_05/stage_05_paired.csv) · '
              '[Trace comparison CSV](stage_05/stage_05_trace_comparisons.csv) · '
              '[Initial states CSV](stage_05/stage_05_initial_boards.csv) · '
              '[Audit manifest](stage_05/final_audit_manifest.json) · '
              '[Raw per-board shards](stage_05/evaluation/shards/) · '
              '[Supervisor manifest](stage_05_supervision/run_manifest.json).', '',
              '## Supervised process times and exact commands', '',
              'The runner elapsed times below use its monotonic clock and include worker startup '
              'and model loading. Per-board times on the figure exclude that one-time work.', '',
              '| Method | State | Completed / planned | Runner elapsed seconds | Peak host RAM % | Peak swap bytes |',
              '|---|---|---:|---:|---:|---:|']
    for row in process_statuses:
        lines.append(f"| {row['method']} | {row['state']} | {row['completed']}/{row['planned']} | "
                     f"{shown(row['elapsed_seconds'])} | {shown(row['peak_host_memory_percent'])} | "
                     f"{display_count(row['peak_swap_used_bytes'])} |")
    lines += ['', 'The supervisor timestamp spans below are separately calculated from UTC start/end '
              'times; clock adjustments can make them differ from monotonic runner elapsed times.', '',
              '| Stage | Status at report snapshot | UTC timestamp span (seconds) |', '|---|---|---:|']
    for row in commands:
        lines.append(f"| {row['stage']} | {row['status_at_report_snapshot']} | {shown(row['process_wall_seconds'])} |")
    lines += ['',
              'Commands and statuses are snapshots taken while this report runs. Consult the final '
              'supervisor manifest for process closure. Existing output paths and absolute time '
              'limits require a fresh output directory and supervision for a new run.', '']
    for row in commands:
        lines += [f"**{row['stage']}** — {row['status_at_report_snapshot']}", '', '```sh', row['command'], '```', '']
    return lines


def aggregate_lines(manifest: Record, history: Record, freeze: Record, summary: list[Record],
                    stage5_status: str, repository: Record, board_manifest: Record | None,
                    inventory: list[Record], qa: Record, prior_audit: Record, contract_audit: Record) -> list[str]:
    resources = manifest.get('resources') or {}
    clock = manifest.get('clock') or {}
    lines = ['# Overnight CaRL campaign — local Sokoban results', '',
             f"Campaign `{manifest.get('campaign_id', 'unknown')}` · global deadline "
             f"`{clock.get('deadline_utc', 'unknown')}` · stage 5 snapshot `{stage5_status}`.", '',
             'Five local experiments are summarised here. Training, inference, BFS and search '
             'use saved local artifacts; this campaign makes no paid/API calls. Stage 4 LoRA training '
             'used the local GPU; policy, rollout, BFS and search workloads use CPU.', '',
             f"WSL resource inventory: {resources.get('logical_affinity_count', 'unknown')} visible "
             f"logical CPUs; {shown((resources.get('memory_total_bytes') or 0) / 2**30, 1)} GiB RAM; "
             f"{resources.get('gpu_observed', 'GPU inventory unavailable')}. CPU pool and thread "
             'settings are stored in each stage’s supervisor and freeze.', '',
             'Boards are grouped by walls and goals modulo rotations/reflections before split '
             'assignment. Multiple trajectory contexts or transformed views from one layout do not '
             'become independent boards. Known historical exposure and the unknown old checkpoint '
             'training split mean the evaluation is exploratory. The ×4 RL and ×10 gamma2/reverse '
             'source distributions differ, so family differences cannot isolate the number of boxes.', '']
    lifecycle = lifecycle_snapshot(manifest)
    if lifecycle['owned_processes_stopped'] is True and lifecycle['supervisor_status'] == 'complete':
        lines += ['Measurement jobs and the campaign watchdog were confirmed stopped before this final '
                  'report render. The render itself ran afterwards as a short local process.', '']
    else:
        lines += [f"At report snapshot the stage lifecycle was `{lifecycle}`; consult the current campaign "
                  'manifest for the final process status.', '']
    lines += history_lines(history)
    lines += [f"Independent audit of stages 1–4: `{prior_audit['state']}`, "
              f"{prior_audit['sources']} named source artifacts; audit SHA256 "
              f"`{prior_audit['sha256'] or 'unavailable'}`. "
              '[Audit details](stage_05/prior_stages_independent_audit.json).', '']
    if prior_audit['issues']:
        lines += [f"Prior-stage audit verification issues: `{prior_audit['issues']}`.", '']
    lines += ['## Stage 5: complete-puzzle full-state baselines', '',
              f"Common per-board cap: {freeze.get('transition_budget', 'not frozen')} simulator transitions "
              f"and {freeze.get('per_job_wall_seconds', 'not frozen')} seconds. The table retains all "
              'planned boards and counts a success only when both caps are met. '
              'Infrastructure gaps and absent jobs are unknown, not behavioural failures; '
              'solved/planned is a lower bound if such gaps exist. A wall timeout is reported separately.', '',
              '| Family | Method | Joint-budget solved / planned | Late solutions | Completed | '
              'Timeouts | Infra / missing | Median plan length |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    if board_manifest:
        lines[-2:-2] = [f"Full puzzle starts were selected before evaluation: "
                        f"{board_manifest.get('selection', 'selection rule unavailable')} "
                        f"Initially solved starts: {board_manifest.get('initially_solved', 'unknown')}. "
                        f"{board_manifest.get('test_cohort', 'Exposure not established.')}", '']
    if inventory:
        lines[-2:-2] = ['Initial boxes already on goals, shown as `count: number of boards`; '
                        'these reflect source distributions rather than selection by model outcome:', '',
                        '| Family | Source | Initial goal placements |', '|---|---|---|',
                        *[f"| {row['family']} | {row['source_family']} | "
                          f"`{row['initial_boxes_on_goals_distribution']}` |" for row in inventory], '']
    for row in summary:
        if row['source_family'] != 'all':
            continue
        lines.append(f"| {row['family']} | {row['method']} | {solved_cell(row)} | "
                     f"{row['solutions_found_after_joint_budget']} | {row['completed']} | "
                     f"{row['timeouts']} | {row['infra_errors']} / {row['missing']} | "
                     f"{shown(row['median_solution_length'], 1)} |")
    lines += ['', '![Complete-puzzle success against transition and time budget](stage_05/stage_05_success.png)', '',
              'Fixed-k kSubS is dashed; dotted lines denote partial results. Overlapping curves show '
              'equal cumulative success fractions along that interval, not a missing method.', '',
              'This is full-state target generation and reaching. The queue’s existing horizons are not '
              'the proposal’s learned relative scale; no Qwen predicate intent is supplied to CLLP. '
              'Equal transition limits do not equalise neural inference cost. The detailed report '
              'separates candidate validity, controller hits, transitions, calls and time.', '',
              '[Detailed stage 5 report](stage_05_report.md) · [Raw board/method records](stage_05/evaluation/shards/) · '
              '[Underlying curves CSV](stage_05/stage_05_curves.csv) · [Frozen configuration](stage_05/evaluation_freeze.json).', '',
              f"Offline replay audit status: `{qa['state']}`; completed audits used "
              f"{display_count(qa['offline_audit_transitions'])} additional simulator transitions, separate from the "
              'planners’ budgets. Other development/early audit calls are itemised in the stage 5 report. '
              f"Independent execution-contract audit: `{contract_audit['state']}`, "
              f"{len(contract_audit['contracts'])} methods; "
              '[details](stage_05/evaluation_contract_audit.json).', '']
    if isinstance(qa.get('interrupted_prior_audit'), dict) and qa['interrupted_prior_audit'].get('offline_transitions') is None:
        lines += ['One earlier interrupted offline audit has an unknown partial transition count; '
                  'it did not stop or fail a model evaluation.', '']
    lines += [
              '## What this campaign does and does not establish', '',
              'The experiments measure full-state reachability, sensitivity of empirical scale to '
              'candidate pools, generator behaviour and a complete-task baseline. They do not show '
              'that policy-relative intent conditioning improves transfer or planning. Generator '
              'validity, a bounded BFS path and learned-controller achievement are distinct outcomes.', '',
              'A useful next experiment is a preregistered paired ablation of horizon banks '
              '`{8}`, `{8,4}`, `{8,1}` and `{8,4,1}` on additional, previously unused boards with '
              'explicitly balanced source strata. Keep CLLP, the value model, branching, and limits '
              'on transitions, time, model calls and tokens common across arms. This tests the role '
              'of the horizon bank, not policy-relative u. In the current data, k=4 use and '
              'kSubS queue exhaustion co-occur in some AdaSubS-only successes; the ablation '
              'would help distinguish this association from a causal contribution.', '',
              '## Repository snapshot', '',
              f"Branch `{repository.get('branch') or 'unknown'}`; HEAD `{repository.get('head') or 'unknown'}`. "
              'These files were changed or untracked at report generation; see the campaign manifest for '
              'the final commit and final status.', '']
    if not repository.get('status_available'):
        lines += ['Git status was unavailable at report generation.', '']
    elif repository.get('changed_files'):
        lines += ['```text', *repository['changed_files'], '```', '']
    else:
        lines += ['Working tree was clean at report generation.', '']
    if repository.get('head_stat'):
        lines += ['Latest commit and its changed-file summary at report generation:', '', '```text',
                  repository['head_stat'], '```', '']
    else:
        lines += ['Latest commit changed-file summary was unavailable at report generation.', '']
    lines += ['[Campaign manifest](campaign_manifest.json) · [Machine-readable summary](campaign_summary.json) · '
              '[Analysis directory](./) · [HTML version](OVERNIGHT_REPORT.html).', '']
    return lines


def html_document(markdown: str) -> str:
    """Produce a small offline HTML document with local linked figure assets."""
    try:
        import mistune

        body = mistune.create_markdown(plugins=['table'])(markdown)
    except ImportError:
        body = '<pre>' + html.escape(markdown) + '</pre>'
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Overnight CaRL campaign</title><style>'
            'body{font:16px/1.5 Georgia,serif;color:#202428;max-width:1200px;margin:2rem auto;padding:0 1rem}'
            'h1,h2,h3{line-height:1.2}table{border-collapse:collapse;font-size:.92rem;margin:1em 0;'
            'display:block;overflow-x:auto}th,td{padding:.35rem .65rem;border-bottom:1px solid #ccc;text-align:left}'
            'img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;'
            'background:#f3f4f5;padding:.8em}a{color:#1c5d88}code{font-size:.9em}'
            '</style></head><body>' + body + '</body></html>\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--boards', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--shards', type=Path)
    parser.add_argument('--merge-manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = (args.output or campaign).resolve()
    stage = output / 'stage_05'
    stage.mkdir(parents=True, exist_ok=True)
    freeze_path = args.freeze or campaign / 'stage_05/evaluation_freeze.json'
    shard_path = args.shards or campaign / 'stage_05/evaluation/shards'
    frozen = optional_json(freeze_path)
    freeze = frozen or {'methods': list(METHODS)}
    boards_path = args.boards or Path(freeze.get('input_boards') or campaign / 'stage_05/boards.jsonl')
    declared = freeze.get('input_boards_sha256')
    board_error = None
    board_manifest_path = boards_path.parent / 'board_manifest.json'
    board_manifest = None
    try:
        boards = read_jsonl(boards_path) if boards_path.is_file() else []
        if declared is not None and (not boards_path.is_file() or sha256(boards_path) != declared):
            raise ValueError('Frozen board file hash differs from the evaluation contract')
        board_manifest = optional_json(board_manifest_path)
        board_manifest_sha = freeze.get('board_manifest_sha256')
        if board_manifest_sha is not None and (not board_manifest_path.is_file()
                                               or sha256(board_manifest_path) != board_manifest_sha):
            raise ValueError('Frozen board manifest hash differs from the evaluation contract')
    except (OSError, ValueError, json.JSONDecodeError) as error:
        boards, board_error = [], f'{type(error).__name__}: {error}'
    merge_path = args.merge_manifest or shard_path.parent / 'merge_manifest.json'
    shard_files, merge = checked_shard_files(shard_path, merge_path)
    plan_error = board_error
    invalid_records = []
    inventory = []
    try:
        if frozen is not None and not board_error:
            inventory = initial_board_inventory(boards)
        cases, invalid_records = (checked_cases(boards, freeze, shard_files)
                                  if frozen is not None and not board_error else ([], []))
    except (ValueError, TypeError, KeyError) as error:
        cases, plan_error = [], f'{type(error).__name__}: {error}'
    summary, curves, pairs = summarize_cases(cases, freeze), success_curves(cases, freeze), paired_comparisons(cases, freeze)
    traces = trace_comparisons(cases, freeze)
    history = dataset_summary(campaign)
    manifest_snapshot = stage / 'report_input_campaign_manifest.json'
    manifest_text = (campaign / 'campaign_manifest.json').read_text(encoding='utf-8')
    manifest = json.loads(manifest_text)
    atomic_text(manifest_snapshot, manifest_text)
    supervisor_path = campaign / 'stage_05_supervision/run_manifest.json'
    supervisor_snapshot = stage / 'report_input_supervisor_manifest.json'
    supervisor = None
    if supervisor_path.is_file():
        supervisor_text = supervisor_path.read_text(encoding='utf-8')
        supervisor = json.loads(supervisor_text)
        atomic_text(supervisor_snapshot, supervisor_text)
    commands = recorded_commands(supervisor)
    process_statuses = evaluation_process_statuses(campaign)
    qa = replay_audit_summary(campaign, freeze)
    historical_audit = prior_stages_audit(campaign)
    contract_audit = evaluation_contract_audit(campaign, freeze_path, freeze)
    repository = repository_snapshot()
    missing = sum(case['result'] is None for case in cases)
    infra = sum(case['result'] is not None and case['result'].get('status') == 'infra_error' for case in cases)
    status = ('pending_freeze' if frozen is None else
              ('complete' if cases and not missing and not infra and not invalid_records and not plan_error
               and merge['state'] == 'complete'
               and merge['hash_verified'] else 'partial'))
    for name, rows in (('stage_05_summary.csv', summary), ('stage_05_curves.csv', curves),
                       ('stage_05_paired.csv', pairs), ('stage_05_trace_comparisons.csv', traces),
                       ('stage_05_initial_boards.csv', inventory)):
        write_csv(stage / name, rows)
    plot_curves(curves, freeze, stage)
    boards_link = os.path.relpath(boards_path.resolve(), output)
    stage_report = '\n'.join(stage_five_lines(freeze, summary, pairs, traces, campaign, commands, process_statuses,
                                                merge, board_manifest, inventory, qa, contract_audit, boards_link,
                                                plan_error, invalid_records)) + '\n'
    aggregate = '\n'.join(aggregate_lines(manifest, history, freeze, summary, status, repository,
                                            board_manifest, inventory, qa, historical_audit, contract_audit)) + '\n'
    atomic_text(output / 'stage_05_report.md', stage_report)
    atomic_text(output / 'OVERNIGHT_REPORT.md', aggregate)
    atomic_text(output / 'OVERNIGHT_REPORT.html', html_document(aggregate))
    inputs = [manifest_snapshot, supervisor_snapshot, campaign / 'stage_01/assessment/assessment.json',
              campaign / 'stage_01/assessment/summary.csv', campaign / 'stage_02_status.json',
              campaign / 'stage_02/analysis_evaluation/rank_summary.csv', campaign / 'stage_03_summary.json',
              campaign / 'stage_04_summary.json', boards_path, board_manifest_path, freeze_path, merge_path,
              campaign / 'stage_05/prior_stages_independent_audit.json',
              campaign / 'stage_05/prior_stages_independent_audit.py',
              campaign / 'stage_05/evaluation_contract_audit.json',
              campaign / 'stage_05/final_audit_manifest.json', campaign / 'stage_05/final_audit_baselines.json',
              campaign / 'stage_05/final_audit_search.json', campaign / 'stage_05/final_audit_baselines_interruption.json',
              *[Path(row['audit_file']) for row in qa['methods'].values() if isinstance(row, dict) and row.get('audit_file')],
              *[Path(row['file']) for row in qa['other_audits']], *shard_files]
    metadata = {'schema_version': 1, 'campaign_id': manifest.get('campaign_id'),
                'stage_05_snapshot': status, 'planned_cases': len(cases),
                'known_cases': sum(known(case) for case in cases), 'missing_cases': missing,
                'infra_error_cases': infra, 'stage_05_freeze': freeze, 'stage_05_merge': merge,
                'stage_05_board_manifest': board_manifest, 'stage_05_plan_error': plan_error,
                'stage_05_invalid_records': invalid_records,
                'stage_05_initial_board_inventory': inventory, 'stage_05_replay_audit': qa,
                'prior_stages_independent_audit': historical_audit,
                'stage_05_evaluation_contract_audit': contract_audit,
                'stage_05_summary': summary,
                'stage_05_paired': pairs, 'stage_05_trace_comparisons': traces,
                'prior_stages': history, 'stage_05_recorded_commands': commands,
                'stage_05_process_statuses': process_statuses, 'repository_at_report': repository,
                'stage_05_lifecycle_at_report': lifecycle_snapshot(manifest),
                'input_sha256': {str(path.resolve()): sha256(path) for path in inputs if path.is_file()},
                'reporter_sha256': sha256(Path(__file__)),
                'limitations': ['Exploratory evaluation boards and unknown checkpoint training exposure',
                                'Source distributions differ between board families',
                                'Full-state baseline is not an intent-conditioned controller',
                                'Missing and infrastructure outcomes are unknown, not behavioural failures']}
    atomic_text(output / 'campaign_summary.json', json.dumps(metadata, ensure_ascii=False, indent=2,
                                                              allow_nan=False) + '\n')
    outputs = ['stage_05_report.md', 'OVERNIGHT_REPORT.md', 'OVERNIGHT_REPORT.html', 'campaign_summary.json',
               'stage_05/stage_05_summary.csv', 'stage_05/stage_05_curves.csv', 'stage_05/stage_05_paired.csv',
               'stage_05/stage_05_trace_comparisons.csv',
               'stage_05/stage_05_initial_boards.csv',
               *[f'stage_05/stage_05_success.{extension}' for extension in ('pdf', 'svg', 'png')]]
    atomic_text(output / 'overnight_report_manifest.json', json.dumps(
        {'schema_version': 1, 'output_sha256': {name: sha256(output / name) for name in outputs},
         'input_sha256': metadata['input_sha256'], 'reporter_sha256': metadata['reporter_sha256'],
         'stage_05_snapshot': status}, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'state': status, 'planned_cases': len(cases), 'known_cases': sum(known(case) for case in cases),
                      'report': str(output / 'OVERNIGHT_REPORT.md')}))


if __name__ == '__main__':
    main()
