"""Stage-three grounding audit for saved or newly generated Sokoban intents.

Mappings are frozen from board identities before any prediction is loaded.
The bounded BFS is a separate local solver, not an intent-conditioned policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.intent_assessment import board_means
from carl.icl.intent_assessment import bootstrap
from carl.icl.intent_assessment import bounded_witness
from carl.icl.intent_assessment import real_replay
from carl.icl.intent_data import predicate_value
from carl.icl.policy_relative_lora import PredictionStatus
from carl.icl.policy_relative_lora import assess_output
from scripts.audit_luna_policy_permutation import derangements

Record = dict[str, Any]
DEFAULT_SEEDS = (20261115, 20261116, 20261117)
DEFAULT_BOOTSTRAP_SEED = 20261115


class Condition(StrEnum):
    ORIGINAL = 'original'
    PERMUTED = 'permuted'
    D4 = 'd4'


class OutputStatus(StrEnum):
    COMPLETE = 'complete'
    MISSING = 'missing'
    UNAVAILABLE = 'permutation_unavailable'


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    max_depth: int = 16
    max_nodes: int = 2500
    seconds: float = 5.0
    bootstrap_draws: int = 2000
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


def sha_file(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_jsonl(path: Path) -> list[Record]:
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temp.open('w', encoding='utf-8') as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def normalize_context(row: Record) -> Record:
    required = ('cohort_id', 'context_id', 'layout_id', 'family', 'source_family', 'split', 'expected_methods')
    if any(key not in row for key in required):
        raise ValueError('Frozen context lacks identity, provenance or declared methods')
    board = row.get('board', row.get('source_rows'))
    if not isinstance(board, list) or not all(isinstance(line, str) for line in board):
        raise ValueError('Frozen context must contain ASCII board rows')
    state = from_rows(board)
    if state.shape != (12, 12, 7) or canonical_layout_id(state) != row['layout_id']:
        raise ValueError('Frozen context has wrong dimensions or D4 layout')
    if not all(character == '#' for character in board[0] + board[-1] + ''.join(r[0] + r[-1] for r in board)):
        raise ValueError('All board boundaries must be walls for the Sokoban simulator')
    methods = row['expected_methods']
    transforms = row.get('expected_transforms', ['identity'])
    if (not isinstance(methods, list) or not methods or len(set(methods)) != len(methods)
            or not isinstance(transforms, list) or not transforms or len(set(transforms)) != len(transforms)
            or transforms[0] != 'identity' or set(transforms) - {'identity', 'rot90_ccw', 'horizontal_reflection'}):
        raise ValueError('Invalid frozen method or D4 transformation list')
    return {**row, 'board': board, 'expected_transforms': transforms}


def freeze_plan(contexts: list[Record], settings: Settings) -> Record:
    """Freeze same-source derangements independently of model outcomes."""
    if (not settings.seeds or len(set(settings.seeds)) != len(settings.seeds) or settings.max_depth < 1
            or settings.max_nodes < 1 or settings.seconds <= 0 or settings.bootstrap_draws < 1):
        raise ValueError('Invalid fixed settings')
    groups: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    identities = set()
    for raw in contexts:
        row = normalize_context(raw)
        ident = (row['cohort_id'], row['context_id'])
        if ident in identities:
            raise ValueError('Duplicate context within a cohort')
        identities.add(ident)
        groups[(row['cohort_id'], row['family'], row['source_family'])].append(row)
    frozen_groups = []
    for (cohort, family, source), rows in sorted(groups.items()):
        rows.sort(key=lambda row: (row['layout_id'], row['context_id']))
        if len({row['layout_id'] for row in rows}) != len(rows):
            raise ValueError('Permutation stratum must contain one context per D4 layout')
        method_sets = {tuple(row['expected_methods']) for row in rows}
        if len(method_sets) != 1:
            raise ValueError('Methods must be predeclared consistently within each stratum')
        assignments = []
        for seed in settings.seeds:
            if len(rows) < 2:
                assignments.append({'seed': seed, 'status': 'unavailable_fewer_than_two_boards', 'mapping': []})
                continue
            indices = derangements(f'{cohort}\0{family}\0{source}', len(rows), seed, 1)[0]
            pairs = [{'target_context_id': rows[target]['context_id'], 'target_layout_id': rows[target]['layout_id'],
                      'donor_context_id': rows[donor]['context_id'], 'donor_layout_id': rows[donor]['layout_id']}
                     for target, donor in enumerate(indices)]
            if any(pair['target_layout_id'] == pair['donor_layout_id'] for pair in pairs):
                raise AssertionError('Permutation returned the original board')
            assignments.append({'seed': seed, 'status': 'complete', 'mapping': pairs})
        frozen_groups.append({'cohort_id': cohort, 'family': family, 'source_family': source,
                              'context_ids': [row['context_id'] for row in rows],
                              'layout_ids': [row['layout_id'] for row in rows],
                              'expected_methods': list(next(iter(method_sets))), 'permutations': assignments})
    return {'schema_version': 1, 'settings': {'seeds': list(settings.seeds), 'max_depth': settings.max_depth,
            'max_nodes': settings.max_nodes, 'safety_seconds': settings.seconds,
            'bootstrap_draws': settings.bootstrap_draws, 'bootstrap_seed': settings.bootstrap_seed},
            'parser': 'strict unique-key JSON and one complete outer Markdown fence removed symmetrically',
            'permutation': 'uniform seeded derangement of original D4 boards within cohort/family/source, '
                           'shared by all methods; invalid and missing move unchanged',
            'groups': frozen_groups}


def prediction_key(row: Record) -> tuple[str, str, str, str]:
    return row['cohort_id'], row['context_id'], row['method'], row.get('transform', 'identity')


def index_predictions(records: list[Record], contexts: list[Record]) -> dict[tuple[str, str, str, str], Record]:
    by_context = {(row['cohort_id'], row['context_id']): normalize_context(row) for row in contexts}
    indexed = {}
    for prediction in records:
        key = prediction_key(prediction)
        context = by_context.get(key[:2])
        if context is None or key in indexed or key[2] not in context['expected_methods'] or key[3] not in context['expected_transforms']:
            raise ValueError(f'Unplanned or duplicate prediction: {key}')
        for field in ('layout_id', 'family', 'source_family', 'split'):
            if prediction.get(field) != context[field]:
                raise ValueError(f'Prediction {field} differs from the frozen context: {key}')
        status = prediction.get('prediction_status')
        raw = prediction.get('raw_output')
        if status not in {'complete', 'missing', 'infra_error', 'not_attempted'}:
            raise ValueError(f'Unknown prediction status: {key}')
        if status == 'complete' and not isinstance(raw, str):
            raise ValueError(f'Completed prediction lacks its raw output: {key}')
        if status != 'complete' and raw is not None:
            raise ValueError(f'Missing or failed prediction contains an output: {key}')
        if key[3] == 'identity' and prediction.get('source_rows') != context['board']:
            raise ValueError(f'Identity prediction refers to a different starting board: {key}')
        indexed[key] = prediction
    return indexed


def make_jobs(contexts: list[Record], plan: Record) -> list[Record]:
    by_id = {(row['cohort_id'], row['context_id']): normalize_context(row) for row in contexts}
    jobs = []
    for key, context in sorted(by_id.items()):
        for method in context['expected_methods']:
            for transform in context['expected_transforms']:
                jobs.append({'cohort_id': key[0], 'context_id': key[1], 'method': method, 'transform': transform,
                             'condition': Condition.ORIGINAL if transform == 'identity' else Condition.D4,
                             'permutation_seed': None, 'donor_context_id': key[1],
                             'donor_layout_id': context['layout_id']})
    for group in plan['groups']:
        for permutation in group['permutations']:
            by_target = {row['target_context_id']: row['donor_context_id'] for row in permutation['mapping']}
            donor_layouts = {row['target_context_id']: row['donor_layout_id'] for row in permutation['mapping']}
            for target in group['context_ids']:
                for method in group['expected_methods']:
                    jobs.append({'cohort_id': group['cohort_id'], 'context_id': target, 'method': method,
                                 'transform': 'identity', 'condition': Condition.PERMUTED,
                                 'permutation_seed': permutation['seed'], 'donor_context_id': by_target.get(target),
                                 'donor_layout_id': donor_layouts.get(target)})
    for job in jobs:
        job['job_id'] = digest(job)
    if len({job['job_id'] for job in jobs}) != len(jobs):
        raise ValueError('Assessment job identities collide')
    return jobs


def canonical_intent(intent: Any) -> str | None:
    if not isinstance(intent, dict) or not all(isinstance(intent.get(key), list) for key in ('effects', 'preserve')):
        return None
    try:
        canonical = {key: sorted(json.dumps(item, sort_keys=True, separators=(',', ':')) for item in intent[key])
                     for key in ('effects', 'preserve')}
        return json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError):
        return None


def is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def support_coordinates(prediction: Record, board: list[str]) -> tuple[set[tuple[int, int]],
                                                                        set[tuple[str, int, int]], str]:
    prompt = prediction.get('prompt')
    if not isinstance(prompt, dict) or not isinstance(prompt.get('messages'), list):
        return set(), set(), 'prompt_not_available'
    examples = [message.get('content') for message in prompt['messages']
                if isinstance(message, dict) and message.get('role') == 'assistant']
    if not examples:
        return set(), set(), 'no_examples'
    if len(examples) != 2 or not all(isinstance(item, str) for item in examples):
        return set(), set(), 'unsupported_example_count_or_type'
    predicates: set[tuple[str, int, int]] = set()
    for example in examples:
        parsed = assess_output(example, board, normalize=True).get('parsed')
        if not isinstance(parsed, dict) or not isinstance(parsed.get('effects'), list):
            return set(), set(), 'support_output_not_parseable'
        for item in parsed['effects']:
            if (isinstance(item, dict) and item.get('predicate') in ('box_at', 'player_at')
                    and is_integer(item.get('row')) and is_integer(item.get('col'))):
                predicates.add((item['predicate'], item['row'], item['col']))
    return {(row, col) for _, row, col in predicates}, predicates, 'two_support_examples'


def intent_diagnostics(intent: Any, board: list[str], errors: list[str], prediction: Record) -> Record:
    goal_count = sum(line.count('.') + line.count('*') + line.count('+') for line in board)
    effects = intent.get('effects') if isinstance(intent, dict) else None
    effects = effects if isinstance(effects, list) else []
    whole_goal = [item for item in effects if isinstance(item, dict)
                  and item.get('predicate') == 'boxes_on_goals_at_least' and item.get('count') == goal_count]
    categories = Counter(item['predicate'] for item in effects
                         if isinstance(item, dict) and isinstance(item.get('predicate'), str))
    coordinate_effects = {(item['predicate'], item['row'], item['col']) for item in effects
                          if isinstance(item, dict) and item.get('predicate') in ('box_at', 'player_at')
                          and is_integer(item.get('row')) and is_integer(item.get('col'))}
    support_positions, support_predicates, support_status = support_coordinates(prediction, board)
    matched_positions = sorted({(row, col) for _, row, col in coordinate_effects} & support_positions)
    matched_predicates = sorted(coordinate_effects & support_predicates)
    truth = []
    for predicate in effects:
        try:
            if not isinstance(predicate, dict) or predicate.get('predicate') not in (
                    'box_at', 'player_at', 'boxes_on_goals_at_least'):
                raise ValueError('unsupported')
            if predicate['predicate'] != 'boxes_on_goals_at_least':
                row, col = predicate.get('row'), predicate.get('col')
                if (set(predicate) != {'predicate', 'row', 'col'} or not is_integer(row) or not is_integer(col)
                        or not (0 <= row < len(board) and 0 <= col < len(board[0])) or board[row][col] == '#'):
                    raise ValueError('out of bounds')
            elif (set(predicate) != {'predicate', 'count'} or not is_integer(predicate.get('count'))
                  or not 1 <= predicate['count'] <= goal_count):
                raise ValueError('invalid count')
            truth.append(predicate_value(predicate, board))
        except (TypeError, ValueError, KeyError, IndexError):
            truth.append(None)
    return {'validation_errors': errors, 'wall_error': any('position is a wall' in error for error in errors),
            'range_error': any('out of range' in error or 'outside valid range' in error for error in errors),
            'any_effect_initially_true': any('effect already true at start' in error for error in errors),
            'all_effects_initially_true': bool(truth) and all(value is True for value in truth),
            'preserve_initially_false': any('preservation false at start' in error for error in errors),
            'effect_predicate_counts': dict(categories), 'effect_count': len(effects),
            'coordinate_effect_count': len(coordinate_effects),
            'requests_whole_goal': bool(whole_goal), 'whole_goal_only': bool(whole_goal) and len(effects) == 1,
            'intent_signature': canonical_intent(intent), 'support_status': support_status,
            'matches_support_coordinate': bool(matched_positions) if support_status == 'two_support_examples' else None,
            'matches_support_predicate_coordinate': bool(matched_predicates) if support_status == 'two_support_examples' else None,
            'matched_support_coordinates': matched_positions if support_status == 'two_support_examples' else None}


def _inverse_replay(board: list[str], intent: Record, actions: list[int], transform: str) -> Record:
    from scripts.overnight_stage03_generate import inverse_intent
    from scripts.overnight_stage03_generate import transform_actions

    mapped, unchanged_instruction = inverse_intent(intent, transform)
    original_actions = transform_actions(actions, transform, inverse=True)
    structured = {**mapped, 'instruction': None} if unchanged_instruction and mapped.get('instruction') else mapped
    reached = real_replay(board, structured, original_actions)
    return {'inverse_intent': mapped, 'inverse_actions': original_actions, 'inverse_replay_verified': True,
            'inverse_reached_observation': reached,
            'instruction_mapping': 'untranslated_natural_instruction' if unchanged_instruction and mapped.get('instruction')
                                   else 'complete'}


def evaluate_job(job: Record, context: Record, prediction: Record | None, settings: Settings,
                 cache: dict[str, tuple[str, Record]]) -> Record:
    transform = job['transform']
    board = context['board']
    if transform != 'identity':
        from scripts.overnight_stage03_generate import transform_rows

        oriented = transform_rows(board, transform)
    else:
        oriented = board
    source_meta = {key: context[key] for key in ('cohort_id', 'context_id', 'layout_id', 'family', 'source_family', 'split')}
    row = {**source_meta, **job,
           'source_rows': oriented, 'start_hash': state_hash(from_rows(oriented)),
           'source_prediction_status': prediction.get('prediction_status') if prediction else None,
           'raw_output': None, 'strict_json_valid': None, 'strict_semantic_valid': None,
           'json_valid': None, 'valid_nontrivial': None, 'fence_removed': None, 'parsed_intent': None,
           'bounded_path_found': None, 'bfs_status': None, 'bfs_search': None, 'bfs_error': None,
           'inverse_replay_verified': None, 'instruction_mapping': None}
    if job['donor_context_id'] is None:
        return {**row, 'output_status': OutputStatus.UNAVAILABLE, 'missing_reason': 'stratum_has_fewer_than_two_layouts'}
    if prediction is None or prediction.get('prediction_status') != 'complete':
        return {**row, 'output_status': OutputStatus.MISSING,
                'missing_reason': (prediction.get('missing_reason') or prediction.get('prediction_status'))
                                  if prediction else 'unexecuted_or_missing_prediction'}
    if transform != 'identity' and prediction.get('source_rows') != oriented:
        raise ValueError('Predicted transformation board differs from the frozen board')
    raw = prediction['raw_output']
    strict = assess_output(raw, oriented, normalize=False)
    normalised = assess_output(raw, oriented, normalize=True)
    intent = normalised.get('parsed')
    errors = normalised['validation_errors']
    row.update({'output_status': OutputStatus.COMPLETE, 'raw_output': raw,
                'strict_json_valid': strict['json_valid'], 'strict_semantic_valid': strict['status'] == PredictionStatus.VALID,
                'json_valid': normalised['json_valid'], 'valid_nontrivial': normalised['status'] == PredictionStatus.VALID,
                'fence_removed': normalised['fence_removed'], 'parsed_intent': intent,
                **intent_diagnostics(intent, oriented, errors, prediction)})
    if transform != 'identity' and row['valid_nontrivial']:
        from scripts.overnight_stage03_generate import inverse_intent

        mapped, unchanged_instruction = inverse_intent(intent, transform)
        row.update({'inverse_intent': mapped, 'inverse_intent_signature': canonical_intent(mapped),
                    'instruction_mapping': 'untranslated_natural_instruction'
                    if unchanged_instruction and mapped.get('instruction') else 'complete'})
    if not row['valid_nontrivial']:
        row['bfs_status'] = 'not_applicable_invalid_intent'
        return row
    cache_key = digest({'board': oriented, 'intent': intent, 'depth': settings.max_depth,
                        'nodes': settings.max_nodes, 'seconds': settings.seconds})
    cached = cache.get(cache_key)
    try:
        if cached is None:
            search = bounded_witness(oriented, intent, max_depth=settings.max_depth,
                                     max_nodes=settings.max_nodes, seconds=settings.seconds)
            if search['status'] != 'time_limit':
                cache[cache_key] = (job['job_id'], search)
            row['bfs_cache_source_job'] = None
            row['bfs_actual_search_seconds'] = search['elapsed_seconds']
        else:
            row['bfs_cache_source_job'], search = cached
            row['bfs_actual_search_seconds'] = 0.0
        row.update({'bfs_search': search, 'bfs_status': search['status'],
                    'bounded_path_found': search['witness_found']})
        if transform != 'identity' and search['witness_found']:
            row.update(_inverse_replay(board, intent, search['witness_actions'], transform))
    except Exception as error:
        row.update({'bfs_status': 'infrastructure_error', 'bfs_error': f'{type(error).__name__}: {error}',
                    'bounded_path_found': None})
    return row


def metric(row: Record, name: str) -> float | None:
    if row.get('output_status') != OutputStatus.COMPLETE:
        return None
    if name == 'valid':
        return float(row['valid_nontrivial'])
    if name == 'bounded_path':
        if not row['valid_nontrivial']:
            return 0.0
        value = row.get('bounded_path_found')
        return float(value) if isinstance(value, bool) else None
    raise ValueError('Unsupported comparison metric')


def interval(rows: list[Record], value: Any, settings: Settings, salt: str) -> Record:
    filtered = [row for row in rows if value(row) is not None]
    values = board_means(filtered, lambda row: value(row))
    material = hashlib.sha256(f'{settings.bootstrap_seed}\0{salt}'.encode()).digest()
    rng = random.Random(int.from_bytes(material[:8], 'big'))
    ci = bootstrap(values, rng, count=settings.bootstrap_draws)
    return {'boards': len(values), 'mean': sum(values) / len(values) if values else None,
            'ci_low': ci[0] if ci else None, 'ci_high': ci[1] if ci else None}


def bootstrap_salt(key: tuple[Any, ...], metric_name: str) -> str:
    """Use the same seed before and after a JSONL resume of StrEnum fields."""
    return json.dumps([*(str(value) for value in key), metric_name], ensure_ascii=False, separators=(',', ':'))


def summarize(rows: list[Record], settings: Settings) -> tuple[list[Record], list[Record]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[Record]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in ('cohort_id', 'family', 'source_family', 'method', 'condition', 'transform'))].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        cohort, family, source, method, condition, transform = key
        complete = [row for row in group if row['output_status'] == OutputStatus.COMPLETE]
        valid = [row for row in complete if row['valid_nontrivial']]
        valid_stats = interval(group, lambda row: metric(row, 'valid'), settings, bootstrap_salt(key, 'valid'))
        bfs_stats = interval(group, lambda row: metric(row, 'bounded_path'), settings, bootstrap_salt(key, 'bfs'))
        signatures = Counter(row['intent_signature'] for row in valid if row.get('intent_signature') is not None)
        most_common, common_count = signatures.most_common(1)[0] if signatures else (None, 0)
        support = [row for row in complete if row.get('support_status') == 'two_support_examples']
        searches = [row for row in valid if row.get('bfs_search') is not None]
        actual = [row for row in searches if row.get('bfs_cache_source_job') is None]
        count_effects = Counter(name for row in valid for name, count in row['effect_predicate_counts'].items()
                                for _ in range(count))
        summaries.append({'cohort_id': cohort, 'family': family, 'source_family': source, 'method': method,
            'condition': condition, 'transform': transform, 'requested_outputs': len(group),
            'requested_layouts': len({row['layout_id'] for row in group}), 'completed_outputs': len(complete),
            'missing_outputs': sum(row['output_status'] == OutputStatus.MISSING for row in group),
            'missing_reasons': dict(Counter(row.get('missing_reason') or 'unspecified' for row in group
                                            if row['output_status'] == OutputStatus.MISSING)),
            'prediction_infrastructure_errors': sum(row.get('source_prediction_status') == 'infra_error' for row in group),
            'unavailable_permutations': sum(row['output_status'] == OutputStatus.UNAVAILABLE for row in group),
            'strict_json_valid': sum(row['strict_json_valid'] for row in complete),
            'strict_semantic_valid': sum(row['strict_semantic_valid'] for row in complete),
            'normalized_json_valid': sum(row['json_valid'] for row in complete),
            'single_fence_removed': sum(row['fence_removed'] for row in complete),
            'valid_nontrivial': len(valid), 'invalid_json': sum(not row['json_valid'] for row in complete),
            'invalid_intent': sum(row['json_valid'] and not row['valid_nontrivial'] for row in complete),
            'wall_errors': sum(row.get('wall_error', False) for row in complete),
            'range_errors': sum(row.get('range_error', False) for row in complete),
            'any_effect_initially_true': sum(row.get('any_effect_initially_true', False) for row in complete),
            'all_effects_initially_true': sum(row.get('all_effects_initially_true', False) for row in complete),
            'preserve_initially_false': sum(row.get('preserve_initially_false', False) for row in complete),
            'whole_goal_requests': sum(row.get('requests_whole_goal', False) for row in complete),
            'whole_goal_only': sum(row.get('whole_goal_only', False) for row in complete),
            'valid_whole_goal_requests': sum(row.get('requests_whole_goal', False) for row in valid),
            'effect_counts_valid': dict(count_effects),
            'conjunction_sizes_valid': dict(Counter(str(row['effect_count']) for row in valid)),
            'distinct_valid_intents': len(signatures), 'most_common_intent': most_common,
            'most_common_intent_count': common_count,
            'most_common_share_valid': common_count / len(valid) if valid else None,
            'two_support_outputs': len(support),
            'two_support_parsed_outputs': sum(row['json_valid'] for row in support),
            'two_support_outputs_with_coordinate_effects': sum(row.get('coordinate_effect_count', 0) > 0 for row in support),
            'matches_support_coordinate': sum(row['matches_support_coordinate'] is True for row in support),
            'matches_support_predicate_coordinate': sum(row['matches_support_predicate_coordinate'] is True for row in support),
            'bounded_paths_found': sum(row.get('bounded_path_found') is True for row in complete),
            'bfs_infrastructure_errors': sum(row.get('bfs_status') == 'infrastructure_error' for row in complete),
            'bfs_status_counts': dict(Counter(row['bfs_status'] for row in valid)),
            'bfs_actual_calls': len(actual), 'bfs_cache_hits': len(searches) - len(actual),
            'bfs_expanded_nodes_actual': sum(row['bfs_search']['expanded_nodes'] for row in actual),
            'bfs_edges_actual': sum(row['bfs_search']['primitive_edges_examined'] for row in actual),
            'bfs_seconds_actual': sum(row['bfs_actual_search_seconds'] for row in actual),
            'valid_board_mean': valid_stats['mean'], 'valid_ci_low': valid_stats['ci_low'],
            'valid_ci_high': valid_stats['ci_high'], 'valid_independent_boards': valid_stats['boards'],
            'bounded_path_board_mean': bfs_stats['mean'], 'bounded_path_ci_low': bfs_stats['ci_low'],
            'bounded_path_ci_high': bfs_stats['ci_high'], 'bounded_path_independent_boards': bfs_stats['boards']})
    base = {(row['cohort_id'], row['context_id'], row['method']): row for row in rows
            if row['condition'] == Condition.ORIGINAL}
    pairs: dict[tuple[str, str, str, str, str, str, str], list[Record]] = defaultdict(list)
    for row in rows:
        if row['condition'] == Condition.ORIGINAL:
            continue
        original = base.get((row['cohort_id'], row['context_id'], row['method']))
        if original is None:
            raise ValueError('Comparison lacks its planned original result')
        for name in ('valid', 'bounded_path'):
            a, b = metric(original, name), metric(row, name)
            if a is not None and b is not None:
                key = (row['cohort_id'], row['family'], row['source_family'], row['method'],
                       row['condition'], row['transform'], name)
                pairs[key].append({'layout_id': row['layout_id'], 'delta': b - a,
                                   'context_id': row['context_id'], 'seed': row['permutation_seed']})
    comparisons = []
    for key, group in sorted(pairs.items()):
        stats = interval(group, lambda row: row['delta'], settings, bootstrap_salt(key, 'paired'))
        comparisons.append({**dict(zip(('cohort_id', 'family', 'source_family', 'method', 'condition', 'transform', 'metric'), key)),
            'paired_observations': len(group), 'paired_contexts': len({row['context_id'] for row in group}),
            'paired_layouts': stats['boards'], 'permutation_seeds': sorted({row['seed'] for row in group if row['seed'] is not None}),
            'mean_comparison_minus_original': stats['mean'], 'ci_low': stats['ci_low'], 'ci_high': stats['ci_high']})
    return summaries, comparisons


def write_csv(path: Path, rows: list[Record]) -> None:
    from io import StringIO

    stream = StringIO(newline='')
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})
    atomic_write(path, stream.getvalue())


def equivariance_summary(rows: list[Record], settings: Settings) -> list[Record]:
    originals = {(row['cohort_id'], row['context_id'], row['method']): row for row in rows
                 if row['condition'] == Condition.ORIGINAL}
    grouped: dict[tuple[str, str, str, str, str], list[Record]] = defaultdict(list)
    for row in rows:
        if row['condition'] != Condition.D4:
            continue
        original = originals[(row['cohort_id'], row['context_id'], row['method'])]
        eligible = (row.get('valid_nontrivial') is True and original.get('valid_nontrivial') is True
                    and row.get('inverse_intent_signature') is not None)
        grouped[(row['cohort_id'], row['family'], row['source_family'], row['method'], row['transform'])].append({
            'layout_id': row['layout_id'], 'eligible': eligible,
            'same_structured': (row['inverse_intent_signature'] == original.get('intent_signature')) if eligible else None,
            'instruction_untranslated': row.get('instruction_mapping') == 'untranslated_natural_instruction'})
    result = []
    for key, group in sorted(grouped.items()):
        stats = interval(group, lambda row: float(row['same_structured']) if row['eligible'] else None,
                         settings, bootstrap_salt(key, 'equivariance'))
        result.append({**dict(zip(('cohort_id', 'family', 'source_family', 'method', 'transform'), key)),
            'planned_contexts': len(group), 'valid_both': sum(row['eligible'] for row in group),
            'instruction_untranslated': sum(row['instruction_untranslated'] for row in group),
            'same_structured_intent': sum(row['same_structured'] is True for row in group),
            'same_structured_board_mean': stats['mean'], 'ci_low': stats['ci_low'], 'ci_high': stats['ci_high'],
            'independent_boards': stats['boards'],
            'limitation': 'structured predicates only; natural-language instruction is not translated'})
    return result


def plot_summaries(output: Path, summaries: list[Record]) -> list[str]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    saved = []
    strata = sorted({(row['cohort_id'], row['family'], row['source_family']) for row in summaries})
    used = set()
    for cohort, family, source in strata:
        from re import sub

        slug = sub(r'[^A-Za-z0-9_.-]+', '_', f'{cohort}_{family}_{source}')
        if slug in used:
            raise ValueError('Cohort labels collide as filenames')
        used.add(slug)
        selected = [row for row in summaries if (row['cohort_id'], row['family'], row['source_family']) ==
                    (cohort, family, source) and row['transform'] == 'identity' and
                    row['condition'] in (Condition.ORIGINAL, Condition.PERMUTED)]
        methods = sorted({row['method'] for row in selected})
        figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True, layout='constrained')
        for axis, metric_name, title in zip(axes, ('valid', 'bounded_path'), ('Valid unsatisfied intent', 'Bounded BFS path')):
            plotted = False
            for index, method in enumerate(methods):
                for condition, color, offset in ((Condition.ORIGINAL, '#21618c', -0.12),
                                                  (Condition.PERMUTED, '#bd5626', 0.12)):
                    candidates = [row for row in selected if row['method'] == method and row['condition'] == condition]
                    if len(candidates) > 1:
                        raise ValueError('Duplicate plot summary for a method and condition')
                    item = candidates[0] if candidates else None
                    value = item.get(f'{metric_name}_board_mean') if item else None
                    if value is None:
                        axis.text(index + offset, 0.04, 'n/a', ha='center', fontsize=7, color=color)
                        continue
                    plotted = True
                    low, high = item.get(f'{metric_name}_ci_low'), item.get(f'{metric_name}_ci_high')
                    x = index + offset
                    if low is not None and high is not None:
                        axis.vlines(x, low, high, color=color, linewidth=1)
                        axis.hlines((low, high), x - 0.03, x + 0.03, color=color, linewidth=1)
                    axis.plot(x, value, 'o', color=color, markersize=4, clip_on=False)
            if not plotted:
                axis.text(.5, .5, 'No completed data', transform=axis.transAxes, ha='center')
            axis.set_title(title)
            axis.set_xticks(range(len(methods)), methods, rotation=18, ha='right')
            axis.set_ylim(-0.03, 1.03)
            axis.set_xlim(-0.5, max(len(methods)-0.5, 0.5))
            axis.grid(axis='y', alpha=.2)
        axes[0].set_ylabel('Fraction with known result, board-weighted')
        figure.suptitle(f'{cohort} · {family} · {source}', fontsize=9)
        figure.legend(handles=[Line2D([], [], marker='o', linestyle='none', color='#21618c', label='Original'),
                               Line2D([], [], marker='o', linestyle='none', color='#bd5626', label='Permuted')],
                      loc='outside lower center', ncol=2, frameon=False, fontsize=8)
        for ext in ('pdf', 'svg', 'png'):
            path = output / f'grounding_{slug}.{ext}'
            temp = output / f'.grounding_{slug}.{ext}.{os.getpid()}.tmp'
            try:
                figure.savefig(temp, format=ext, dpi=180, bbox_inches='tight')
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
            saved.append(path.name)
        plt.close(figure)
    return saved


def fresh_contexts(run: Path) -> tuple[list[Record], list[Record], list[Record], dict[str, str]]:
    """Read a generator's frozen plan; prediction bytes are loaded separately."""
    plan_path, contract_path, prompts_path = (run / name for name in ('plan.json', 'contract.json', 'prompts.jsonl'))
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    prompts = read_jsonl(prompts_path)
    if digest(plan) != contract['plan_sha256'] or digest(prompts) != contract['prompts_sha256']:
        raise ValueError('Fresh generator plan or prompts differ from their contract')
    cases = plan['cases']
    if len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('Fresh generator case IDs are duplicated')
    boards: dict[tuple[str, str], list[str]] = {}
    for case in cases:
        key = (case['cohort_id'], case['context_id'])
        if boards.setdefault(key, case['original_rows']) != case['original_rows']:
            raise ValueError('Fresh generator cases disagree on original board')
    contexts = []
    for row in plan['context_manifest']:
        key = (row['cohort_id'], row['context_id'])
        if key not in boards:
            raise ValueError('Fresh generator context is absent from its frozen cases')
        contexts.append({**row, 'board': boards[key]})
    identities = {(row['cohort_id'], row['context_id']) for row in contexts}
    if identities != set(boards):
        raise ValueError('Fresh generator cases and context manifest differ')
    prompt_map = {row['case_id']: row for row in prompts}
    if len(prompt_map) != len(prompts) or set(prompt_map) != {case['case_id'] for case in cases}:
        raise ValueError('Fresh generator prompt coverage differs from its case plan')
    return contexts, cases, prompts, {str(path.resolve()): sha_file(path) for path in (plan_path, contract_path, prompts_path)}


def load_fresh_predictions(run: Path, cases: list[Record], prompts: list[Record]) -> list[Record]:
    by_case = {row['case_id']: row for row in cases}
    prompt_map = {row['case_id']: row for row in prompts}
    path = run / 'predictions.jsonl'
    rows = read_jsonl(path) if path.exists() else []
    contract = json.loads((run / 'contract.json').read_text(encoding='utf-8'))
    expected_contract = digest(contract)
    for row in rows:
        case = by_case.get(row.get('case_id'))
        if case is None or row.get('contract_sha256') != expected_contract:
            raise ValueError('Unplanned fresh output or incorrect contract')
        if any(row.get(key) != case.get(key) for key in ('cohort_id', 'context_id', 'layout_id', 'family',
                                                         'source_family', 'split', 'method', 'transform',
                                                         'original_rows', 'source_rows', 'prompt_sha256')):
            raise ValueError('Fresh output disagrees with its frozen case')
        row['prompt'] = prompt_map[row['case_id']]
    return rows


def check_multisets(plan: Record, indexed: dict[tuple[str, str, str, str], Record]) -> list[Record]:
    checks = []
    for group in plan['groups']:
        cohort, ids = group['cohort_id'], group['context_ids']
        for method in group['expected_methods']:
            def token(context: str) -> str:
                row = indexed.get((cohort, context, method, 'identity'))
                return json.dumps([row.get('prediction_status'), row.get('raw_output')] if row else ['missing', None],
                                  ensure_ascii=False, sort_keys=True)
            before = Counter(token(context) for context in ids)
            for permutation in group['permutations']:
                if permutation['status'] != 'complete':
                    checks.append({'cohort_id': cohort, 'family': group['family'], 'source_family': group['source_family'],
                                   'method': method, 'seed': permutation['seed'], 'status': permutation['status']})
                    continue
                after = Counter(token(item['donor_context_id']) for item in permutation['mapping'])
                if before != after:
                    raise AssertionError('Permutation altered raw-output frequencies')
                checks.append({'cohort_id': cohort, 'family': group['family'], 'source_family': group['source_family'],
                               'method': method, 'seed': permutation['seed'], 'status': 'frequency_preserved',
                               'outputs': len(ids), 'distinct_raw_status_pairs': len(before)})
    return checks


def recover_rows(path: Path, jobs: list[Record], contexts: dict[tuple[str, str], Record],
                 indexed: dict[tuple[str, str, str, str], Record]) -> list[Record]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        prefix, _, tail = raw.rpartition(b'\n')
        backup = path.with_name(f'{path.name}.partial-{hashlib.sha256(tail).hexdigest()[:12]}')
        backup.write_bytes(tail)
        atomic_write(path, prefix.decode('utf-8') + ('\n' if prefix else ''))
    rows = read_jsonl(path)
    if len(rows) > len(jobs) or any(row['job_id'] != job['job_id'] for row, job in zip(rows, jobs)):
        raise ValueError('Saved assessments do not match frozen job order')
    for row, job in zip(rows, jobs):
        if any(row.get(key) != value for key, value in job.items()):
            raise ValueError('Saved assessment has changed frozen job fields')
        expected_digest = digest({key: value for key, value in row.items() if key != 'record_sha256'})
        if row.get('record_sha256') != expected_digest:
            raise ValueError('Saved assessment record digest differs')
        context = contexts[(job['cohort_id'], job['context_id'])]
        for field in ('cohort_id', 'context_id', 'layout_id', 'family', 'source_family', 'split'):
            if row.get(field) != context[field]:
                raise ValueError('Saved assessment has changed target context')
        prediction = indexed.get((job['cohort_id'], job['donor_context_id'], job['method'], job['transform']))
        if job['donor_context_id'] is None:
            expected_status, expected_raw = OutputStatus.UNAVAILABLE, None
        elif prediction is None or prediction.get('prediction_status') != 'complete':
            expected_status, expected_raw = OutputStatus.MISSING, None
        else:
            expected_status, expected_raw = OutputStatus.COMPLETE, prediction['raw_output']
        if row.get('output_status') != expected_status or row.get('raw_output') != expected_raw:
            raise ValueError('Saved assessment no longer matches its donor raw output')
    return rows


def report(output: Path, rows: list[Record], summaries: list[Record], comparisons: list[Record],
           equivariance: list[Record], settings: Settings) -> None:
    complete = sum(row['output_status'] == OutputStatus.COMPLETE for row in rows)
    missing = sum(row['output_status'] == OutputStatus.MISSING for row in rows)
    unavailable = sum(row['output_status'] == OutputStatus.UNAVAILABLE for row in rows)
    timeout = sum(row.get('bfs_status') == 'time_limit' for row in rows)
    errors = sum(row.get('bfs_status') == 'infrastructure_error' for row in rows)
    lines = ['# Sokoban intent grounding audit', '',
        f'Completed outputs: {complete}; missing outputs: {missing}; unavailable permutation slots: {unavailable}.',
        'Primary parsing removes at most one complete outer Markdown fence uniformly; strict JSON and strict semantic '
        'validity are reported separately. Duplicate JSON keys are rejected. Each requested effect must be initially false.',
        f'Bounded BFS: at most {settings.max_depth} primitive actions and {settings.max_nodes} expanded nodes; '
        f'{settings.seconds:g} s safety timeout per fresh search. Time cutoffs: {timeout}; '
        f'solver/replay infrastructure errors: {errors}. A cutoff means unresolved, not impossible.',
        'All action paths found by BFS are replayed in the real Sokoban environment. This is a bounded solver, '
        'not an intent-conditioned controller or a complete-task planning test.', '',
        'Permutations move complete raw outputs between distinct boards only within the same cohort, '
        'board family and source distribution. The same mappings are applied to every method, with invalid and '
        'missing outputs retained. Cohorts remain separate even when their layouts overlap.',
        'The permuted requested-output count repeats the same saved responses for each of the three fixed seeds; '
        'it is not a count of newly generated responses. Bootstrap intervals condition on these frozen mappings.',
        'Validity, solver outcomes, coordinate copying, whole-goal requests, conjunction sizes and full '
        'per-output records are available in the CSV/JSONL files. Bootstrap resamples canonical layouts, '
        'not contexts, transforms or permutation draws; fewer than five boards receive no interval.',
        'Transformed outputs are mapped to original coordinates for structured intent comparisons. '
        'Natural-language instruction text is explicitly marked untranslated; action-path inverse replay '
        'checks structured predicates only.', '',
        '| Cohort | Family | Source | Method | Pairing | Complete / requested | Valid / completed | BFS paths / completed |',
        '|---|---|---|---|---|---:|---:|---:|']
    for row in summaries:
        if row['transform'] != 'identity':
            continue
        lines.append(f"| {row['cohort_id']} | {row['family']} | {row['source_family']} | {row['method']} | "
                     f"{row['condition']} | {row['completed_outputs']}/{row['requested_outputs']} | "
                     f"{row['valid_nontrivial']}/{row['completed_outputs']} | "
                     f"{row['bounded_paths_found']}/{row['completed_outputs']} |")
    lines += ['', 'Missing output and infrastructure failures are reported separately from semantic invalidity. '
              'For bounded paths, a solver infrastructure error has unknown outcome; the displayed numerator is '
              'therefore a lower bound whenever such an error occurs. Source-family differences and historical '
              'checkpoint exposure prevent causal claims about board size, transfer or policy-relative planning.',
              '', 'See `paired_comparisons.csv` for paired layout-bootstrap comparisons and '
              '`equivariance_summary.csv` for exact structured-intent agreement across D4 transforms.']
    atomic_write(output / 'report.md', '\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contexts', type=Path, nargs='*', default=[])
    parser.add_argument('--predictions', type=Path, nargs='*', default=[])
    parser.add_argument('--generator-run', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--max-depth', type=int, default=16)
    parser.add_argument('--max-nodes', type=int, default=2500)
    parser.add_argument('--seconds', type=float, default=5.0)
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.contexts and not args.generator_run:
        parser.error('Supply frozen contexts and predictions, or a fresh generator run')
    settings = Settings(tuple(args.seeds), args.max_depth, args.max_nodes, args.seconds,
                        args.bootstrap_draws, args.bootstrap_seed)
    contexts = [row for path in args.contexts for row in read_jsonl(path)]
    fresh = []
    input_hashes = {str(path.resolve()): sha_file(path) for path in (*args.contexts, *args.predictions)}
    for run in args.generator_run:
        fresh_context, cases, prompts, hashes = fresh_contexts(run)
        contexts.extend(fresh_context)
        fresh.append((run, cases, prompts))
        input_hashes.update(hashes)
        path = run / 'predictions.jsonl'
        input_hashes[str(path.resolve())] = sha_file(path) if path.is_file() else None
    plan = freeze_plan(contexts, settings)
    jobs = make_jobs(contexts, plan)
    protocol = {'schema_version': 1, 'plan_sha256': digest(plan), 'inputs_sha256': input_hashes,
                'assessor_sha256': sha_file(Path(__file__)), 'argv': sys.argv,
                'solver': {'max_depth': settings.max_depth, 'max_nodes': settings.max_nodes,
                           'safety_seconds': settings.seconds},
                'interpretation': ['BFS finds bounded paths; it is not a learned policy.',
                    'Permutation distributions condition on the same raw outputs and do not establish independence.',
                    'Timeouts are recorded separately and make node-limited comparisons timing-dependent.',
                    'Historical cohorts were already inspected and are exploratory.']}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plan_path, protocol_path = output / 'permutations.json', output / 'protocol.json'
    if plan_path.exists() or protocol_path.exists():
        if not args.resume or not plan_path.is_file() or not protocol_path.is_file():
            raise ValueError('Existing assessment requires --resume and both frozen files')
        old = json.loads(protocol_path.read_text(encoding='utf-8'))
        if {key: value for key, value in old.items() if key != 'argv'} != {
                key: value for key, value in protocol.items() if key != 'argv'} or digest(
                    json.loads(plan_path.read_text(encoding='utf-8'))) != digest(plan):
            raise ValueError('Inputs or settings changed; use a new assessment directory')
    else:
        atomic_write(plan_path, json.dumps(plan, indent=2, sort_keys=True) + '\n')
        atomic_write(protocol_path, json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    # No raw response is parsed or used until the entire board-only plan is durable.
    predictions = [row for path in args.predictions for row in read_jsonl(path)]
    for run, cases, prompts in fresh:
        predictions.extend(load_fresh_predictions(run, cases, prompts))
    for path_name, expected in input_hashes.items():
        path = Path(path_name)
        if (sha_file(path) if path.is_file() else None) != expected:
            raise ValueError('An input changed after the frozen snapshot')
    indexed = index_predictions(predictions, contexts)
    checks = check_multisets(plan, indexed)
    atomic_write(output / 'permutation_checks.json', json.dumps(checks, indent=2, sort_keys=True) + '\n')
    by_context = {(row['cohort_id'], row['context_id']): normalize_context(row) for row in contexts}
    results_path = output / 'assessments.jsonl'
    if not args.resume and results_path.exists() and results_path.stat().st_size:
        raise FileExistsError('Assessment output already exists; use --resume')
    rows = recover_rows(results_path, jobs, by_context, indexed) if args.resume else []
    cache: dict[str, tuple[str, Record]] = {}
    for row in rows:
        if row.get('valid_nontrivial') and row.get('bfs_search') and row['bfs_search']['status'] != 'time_limit':
            key = digest({'board': row['source_rows'], 'intent': row['parsed_intent'], 'depth': settings.max_depth,
                          'nodes': settings.max_nodes, 'seconds': settings.seconds})
            cache.setdefault(key, (row.get('bfs_cache_source_job') or row['job_id'], row['bfs_search']))
    atomic_write(output / 'status.json', json.dumps({'state': 'running', 'completed_jobs': len(rows),
                                                    'planned_jobs': len(jobs)}, indent=2) + '\n')
    started = time.monotonic()
    try:
        with results_path.open('a', encoding='utf-8', buffering=1) as destination:
            for job in jobs[len(rows):]:
                context = by_context[(job['cohort_id'], job['context_id'])]
                key = (job['cohort_id'], job['donor_context_id'], job['method'], job['transform'])
                result = evaluate_job(job, context, indexed.get(key), settings, cache)
                result['record_sha256'] = digest(result)
                destination.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                destination.flush()
                os.fsync(destination.fileno())
                rows.append(result)
                if len(rows) % 50 == 0:
                    atomic_write(output / 'status.json', json.dumps({'state': 'running', 'completed_jobs': len(rows),
                        'planned_jobs': len(jobs), 'elapsed_seconds': time.monotonic() - started,
                        'output_statuses': dict(Counter(row['output_status'] for row in rows))}, indent=2) + '\n')
        summaries, comparisons = summarize(rows, settings)
        equivariance = equivariance_summary(rows, settings)
        for name, data in (('assessments.csv', rows), ('summary.csv', summaries),
                           ('paired_comparisons.csv', comparisons), ('equivariance_summary.csv', equivariance)):
            write_csv(output / name, data)
        atomic_write(output / 'summary.json', json.dumps({'summary': summaries, 'comparisons': comparisons,
            'equivariance': equivariance}, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        figures = plot_summaries(output, summaries)
        report(output, rows, summaries, comparisons, equivariance, settings)
        filenames = ['permutations.json', 'protocol.json', 'permutation_checks.json', 'assessments.jsonl',
                     'assessments.csv', 'summary.csv', 'paired_comparisons.csv', 'equivariance_summary.csv',
                     'summary.json', 'report.md', *figures]
        manifest = {'schema_version': 1, 'plan_sha256': digest(plan), 'inputs_sha256': input_hashes,
                    'outputs_sha256': {name: sha_file(output / name) for name in filenames},
                    'finished_jobs': len(rows), 'planned_jobs': len(jobs), 'elapsed_seconds': time.monotonic() - started,
                    'timeouts': sum(row.get('bfs_status') == 'time_limit' for row in rows),
                    'solver_infrastructure_errors': sum(row.get('bfs_status') == 'infrastructure_error' for row in rows)}
        atomic_write(output / 'assessment_manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        atomic_write(output / 'status.json', json.dumps({'state': 'complete', **manifest}, indent=2) + '\n')
    except BaseException as error:
        atomic_write(output / 'status.json', json.dumps({'state': 'failed', 'completed_jobs': len(rows),
            'planned_jobs': len(jobs), 'elapsed_seconds': time.monotonic() - started,
            'error': f'{type(error).__name__}: {error}'}, indent=2) + '\n')
        raise
    print(json.dumps({'output': str(output.resolve()), 'completed': len(rows), 'planned': len(jobs),
                      'timeouts': manifest['timeouts'], 'solver_errors': manifest['solver_infrastructure_errors']}))


if __name__ == '__main__':
    main()
