#!/usr/bin/env python3
"""Prepare full-state candidate pools from the frozen overnight board splits.

Demonstration targets are oracle targets, not outputs of a deployable
generator. BFS enumerates simulator-reachable states without consulting any
controller results. Every chosen path is independently replayed in CaRL.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_assessment import State
from carl.icl.intent_assessment import board_parts
from carl.icl.intent_assessment import step

DEMO_OFFSETS = (1, 4, 16, 32, 8, 2)
BFS_SLOTS = (
    'bfs_d01_04_p0', 'bfs_d01_04_p1plus',
    'bfs_d05_08_p0', 'bfs_d05_08_p1plus',
    'bfs_d09_max_p0', 'bfs_d09_max_p1plus',
)
DEMO_SLOTS = tuple(f'demo_{index:02d}' for index in range(1, 5))
CONTEXT_FIELDS = ('context_id', 'layout_id', 'family', 'source_family', 'split', 'board_rank',
                  'context_rank', 'source', 'source_hash')
PROVENANCE_FIELDS = ('source_path', 'source_sha256', 'trajectory_id', 'i', 'original_split', 'exposure_status')


class Origin(StrEnum):
    DEMONSTRATION = 'demonstration'
    BFS = 'bfs'
    GENERATOR = 'generator'


class PoolVariant(StrEnum):
    COMPLETE = 'demo4_bfs6'
    FALLBACK = 'demo4_bfs4'
    UNDERFILLED = 'underfilled'


@dataclass(frozen=True)
class Node:
    parent: State | None
    action: int | None
    depth: int
    pushes: int


def digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def candidate_id(context_id: str, target_hash: str) -> str:
    return digest({'context_id': context_id, 'target_hash': target_hash})


def replay(source: list[str], actions: list[int]) -> list[str]:
    """Replay a fragment, rejecting no-ops and actions after task completion."""
    current = from_rows(source)
    env = environment(current)
    for action in actions:
        if isinstance(action, bool) or action not in range(4) or env.is_solved(current):
            raise ValueError('Invalid action or fragment continues after task completion')
        reached = np.asarray(env.next_state(current, action))
        validate_subgoal(current, reached)
        if np.array_equal(current, reached):
            raise ValueError('Fragment contains a no-op')
        current = reached
    return to_rows(current)


def rows_for_state(source: list[str], state: State) -> list[str]:
    height, width, walls, goals, _, _ = board_parts(source)
    player, boxes = state
    rows = [['#' if (r, c) in walls else '.' if (r, c) in goals else '-'
             for c in range(width)] for r in range(height)]
    for r, c in boxes:
        rows[r][c] = '*' if (r, c) in goals else '$'
    r, c = player
    rows[r][c] = '+' if (r, c) in goals else '@'
    return [''.join(row) for row in rows]


def bfs_slot(node: Node) -> str:
    depth = 'd01_04' if node.depth <= 4 else 'd05_08' if node.depth <= 8 else 'd09_max'
    pushes = 'p0' if node.pushes == 0 else 'p1plus'
    return f'bfs_{depth}_{pushes}'


def bfs_states(source: list[str], max_depth: int, max_nodes: int) -> tuple[dict[State, Node], dict[str, Any]]:
    """One deterministic BFS per context; the node limit is shared by all slots."""
    height, width, walls, goals, boxes, player = board_parts(source)
    initial = (player, boxes)
    nodes = {initial: Node(None, None, 0, 0)}
    frontier = deque([initial])
    expanded = edges = 0
    depth_limited = False
    start = time.monotonic()
    while frontier and expanded < max_nodes:
        current = frontier.popleft()
        node = nodes[current]
        if node.depth >= max_depth:
            depth_limited = True
            continue
        if current[1] <= goals:
            continue
        expanded += 1
        for action in range(4):
            edges += 1
            successor = step(*current, action, height, width, walls)
            if successor is None or successor in nodes:
                continue
            nodes[successor] = Node(current, action, node.depth + 1,
                                    node.pushes + int(successor[1] != current[1]))
            frontier.append(successor)
    status = 'node_limit' if frontier and expanded >= max_nodes else 'depth_limit' if depth_limited else 'exhausted'
    return nodes, {'status': status, 'max_depth': max_depth, 'max_nodes': max_nodes,
                   'expanded_nodes': expanded, 'generated_states': len(nodes), 'edges_examined': edges,
                   'elapsed_seconds': time.monotonic() - start}


def path_to(nodes: dict[State, Node], state: State) -> list[int]:
    actions: list[int] = []
    while nodes[state].parent is not None:
        node = nodes[state]
        assert node.action is not None and node.parent is not None
        actions.append(node.action)
        state = node.parent
    return list(reversed(actions))


def select_pool_variant(demo_count: int, available_bfs: set[str]) -> tuple[PoolVariant, set[str]]:
    """Keep proportions fixed for complete pools; label all smaller pools."""
    if not available_bfs <= set(BFS_SLOTS):
        raise ValueError('Unknown BFS slot')
    if demo_count == 4 and len(available_bfs) == 6:
        return PoolVariant.COMPLETE, available_bfs
    if demo_count == 4 and len(available_bfs) >= 4:
        return PoolVariant.FALLBACK, set([slot for slot in BFS_SLOTS if slot in available_bfs][:4])
    return PoolVariant.UNDERFILLED, available_bfs


def make_record(context: dict[str, Any], target: list[str], origin: Origin) -> dict[str, Any]:
    target_hash = state_hash(from_rows(target))
    return {**{key: context[key] for key in CONTEXT_FIELDS},
            **{key: context.get(key) for key in PROVENANCE_FIELDS},
            'candidate_id': candidate_id(context['context_id'], target_hash), 'target': target,
            'target_hash': target_hash, 'candidate_origin': origin, 'demo_offsets': [], 'demo_pair_ids': [],
            'bfs_depth': None, 'bfs_pushes': None, 'bfs_actions': None,
            'in_fixed_pool': False, 'in_generator_pool': False, 'pool_variant': None, 'slot': None}


def prepare_context(pairs: list[dict[str, Any]], max_depth: int, max_nodes: int, seed: int) -> dict[str, Any]:
    """Build all demo candidates and the fixed, controller-independent pool."""
    if not pairs:
        raise ValueError('A context requires at least one demonstration pair')
    context = pairs[0]
    source = context['source']
    initial = from_rows(source)
    if state_hash(initial) != context['source_hash'] or canonical_layout_id(initial) != context['layout_id']:
        raise ValueError('Source hash or canonical layout differs from frozen input')
    by_target: dict[str, dict[str, Any]] = {}
    demo_by_offset: dict[int, dict[str, Any]] = {}
    demo_states: set[State] = set()
    for pair in sorted(pairs, key=lambda row: row['k']):
        if any(pair.get(key) != context.get(key) for key in CONTEXT_FIELDS):
            raise ValueError('Pairs in one context disagree on metadata or current board')
        offset = pair['k']
        if offset in demo_by_offset or not isinstance(offset, int) or offset < 1:
            raise ValueError('Duplicate or invalid demonstration offset')
        target = pair['target']
        if len(pair['demonstration_actions']) != offset or replay(source, pair['demonstration_actions']) != target:
            raise ValueError('Recorded demonstration does not replay to its full target')
        target_board = from_rows(target)
        validate_subgoal(initial, target_board)
        target_hash = state_hash(target_board)
        if target_hash != pair['target_hash'] or target_hash == context['source_hash']:
            raise ValueError('Demonstration target hash is invalid or already satisfied')
        record = by_target.setdefault(target_hash, make_record(context, target, Origin.DEMONSTRATION))
        if record['target'] != target:
            raise ValueError('Target hash collision')
        record['demo_offsets'].append(offset)
        record['demo_pair_ids'].append(pair['pair_id'])
        demo_by_offset[offset] = record
        _, _, _, _, boxes, player = board_parts(target)
        demo_states.add((player, boxes))

    nodes, bfs = bfs_states(source, max_depth, max_nodes)
    choices: dict[str, list[State]] = {slot: [] for slot in BFS_SLOTS}
    for state, node in nodes.items():
        if node.depth and state not in demo_states:
            choices[bfs_slot(node)].append(state)
    selected_bfs: dict[str, dict[str, Any]] = {}
    for slot in BFS_SLOTS:
        options = choices[slot]
        if not options:
            continue
        def choice_key(state: State) -> str:
            return digest([seed, context['context_id'], slot, state[0], sorted(state[1])])
        chosen = min(options, key=choice_key)
        node = nodes[chosen]
        actions = path_to(nodes, chosen)
        target = rows_for_state(source, chosen)
        if replay(source, actions) != target or len(actions) != node.depth:
            raise ValueError('BFS path differs from real environment replay')
        record = make_record(context, target, Origin.BFS)
        if record['target_hash'] in by_target:
            raise ValueError('BFS endpoint duplicates a demonstration')
        record.update(bfs_depth=node.depth, bfs_pushes=node.pushes, bfs_actions=actions)
        by_target[record['target_hash']] = record
        selected_bfs[slot] = record
    bfs['slot_available_counts'] = {key: len(value) for key, value in choices.items()}

    slots: list[dict[str, Any]] = []
    offset_availability: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for offset in DEMO_OFFSETS:
        record = demo_by_offset.get(offset)
        if record is None:
            status = 'not_available'
        elif record['candidate_id'] in assigned:
            status = 'duplicate_target'
        elif len(slots) >= len(DEMO_SLOTS):
            status = 'outside_first_four'
        else:
            status = 'selected'
            assigned.add(record['candidate_id'])
            slots.append({'slot': DEMO_SLOTS[len(slots)], 'candidate_id': record['candidate_id'],
                          'status': 'available', 'reason': None, 'selected_offset': offset})
        offset_availability.append({'offset': offset, 'status': status,
                                    'candidate_id': None if record is None else record['candidate_id']})
    while len(slots) < len(DEMO_SLOTS):
        slots.append({'slot': DEMO_SLOTS[len(slots)], 'candidate_id': None, 'status': 'missing',
                      'reason': 'insufficient_distinct_demo_targets', 'selected_offset': None})
    demo_count = len(assigned)
    variant, chosen_bfs = select_pool_variant(demo_count, set(selected_bfs))
    for slot in BFS_SLOTS:
        record = selected_bfs.get(slot)
        status = 'missing' if record is None else 'available' if slot in chosen_bfs else 'excluded_by_fallback'
        reason = 'no_endpoint_within_limits' if record is None else 'fallback_uses_first_four' if status != 'available' else None
        slots.append({'slot': slot, 'candidate_id': None if record is None else record['candidate_id'],
                      'status': status, 'reason': reason, 'selected_offset': None})
    records = {record['candidate_id']: record for record in by_target.values()}
    for item in slots:
        if item['status'] == 'available':
            record = records[item['candidate_id']]
            record.update(in_fixed_pool=True, pool_variant=variant, slot=item['slot'])
            item['status'] = 'selected'
    summary = {**{key: context[key] for key in CONTEXT_FIELDS if key not in ('source',)},
               'status': 'complete', 'pool_variant': variant, 'slots': slots, 'bfs': bfs,
               'demo_offset_availability': offset_availability, 'demo_candidate_count': len(demo_states),
               'fixed_candidate_count': sum(r['in_fixed_pool'] for r in records.values())}
    return {'context': summary, 'candidates': sorted(records.values(), key=lambda row: row['candidate_id'])}


def select_contexts(pairs: list[dict[str, Any]], split: str, max_boards: int | None,
                    max_contexts: int, exposure: dict[str, Any]) -> list[list[dict[str, Any]]]:
    protected = set(exposure['training_prohibited_layout_ids'])
    evaluation = set(exposure['campaign_evaluation_layout_ids'])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        if pair['split'] != split or pair['context_rank'] >= max_contexts:
            continue
        if max_boards is not None and pair['board_rank'] >= max_boards:
            continue
        layout = pair['layout_id']
        if split == 'evaluation' and (layout not in evaluation or layout not in protected):
            raise ValueError('Evaluation layout is missing its campaign reservation')
        if split == 'development' and layout in protected:
            raise ValueError('Protected layout cannot be used for development')
        grouped.setdefault(pair['context_id'], []).append(pair)
    selected = sorted(grouped.values(), key=lambda group: (group[0]['board_rank'], group[0]['family'],
                                                           group[0]['context_rank']))
    if not selected:
        raise ValueError('No contexts selected')
    return selected


def work(pairs: list[dict[str, Any]], max_depth: int, max_nodes: int, seed: int) -> dict[str, Any]:
    try:
        return prepare_context(pairs, max_depth, max_nodes, seed)
    except Exception as error:
        return {'context': {**{key: pairs[0][key] for key in CONTEXT_FIELDS if key != 'source'},
                            'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                            'pool_variant': None, 'slots': [], 'bfs': None}, 'candidates': []}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_depth < 9 or args.max_nodes < 1 or args.workers < 1 or args.max_contexts not in (1, 2):
        raise ValueError('max-depth >= 9, positive nodes/workers and one or two contexts are required')
    if args.workers > len(os.sched_getaffinity(0)):
        raise ValueError('workers exceed available CPU affinity')
    if args.max_boards is not None and args.max_boards < 1:
        raise ValueError('max-boards must be positive')
    source_dir, output = args.input_dir.resolve(), args.output.resolve()
    source_manifest = json.loads((source_dir / 'data_manifest.json').read_text())
    for name in ('pairs.jsonl', 'exposure_register.json'):
        if file_sha(source_dir / name) != source_manifest['outputs_sha256'][name]:
            raise ValueError(f'Frozen stage01 data differ from their manifest: {name}')
    exposure = json.loads((source_dir / 'exposure_register.json').read_text())
    groups = select_contexts(read_jsonl(source_dir / 'pairs.jsonl'), args.split, args.max_boards,
                             args.max_contexts, exposure)
    contract = {'schema_version': 1, 'input_dir': str(source_dir),
                'input_manifest_sha256': file_sha(source_dir / 'data_manifest.json'),
                'preparation_sha256': file_sha(Path(__file__)), 'split': args.split,
                'max_boards_per_family': args.max_boards, 'max_contexts_per_board': args.max_contexts,
                'max_depth': args.max_depth, 'max_nodes_per_context': args.max_nodes, 'seed': args.seed,
                'demo_slots': list(DEMO_SLOTS), 'demo_offset_priority': list(DEMO_OFFSETS), 'bfs_slots': list(BFS_SLOTS),
                'fallback_rule': 'four distinct demo and first four available BFS slots when four or five BFS slots '
                                 'exist; otherwise underfilled; no controller outcomes used',
                'demo_selection_rule': 'take the first four distinct available demo targets in offset-priority order; '
                                       'retain all available offsets and deduplicated targets',
                'sampling_rule': 'minimum SHA256 of seed, context, slot, player and sorted box positions',
                'planned_context_ids': [group[0]['context_id'] for group in groups],
                'roles': {'train': {'boards': 0, 'reason': 'no verified separate unreserved training pool in input'},
                          'development': 'original stage01 development, benchmark only',
                          'evaluation': 'original stage01 evaluation, evaluator only; no training or tuning'},
                'limitations': ['Demo future states are oracle candidates, not generator outputs.',
                                'BFS confirms a path, not successful execution by a learned controller.',
                                'Push counts are for the chosen shortest BFS path, not necessarily minimum pushes.']}
    contract_path = output / 'contract.json'
    if contract_path.exists():
        if not args.resume or json.loads(contract_path.read_text()) != contract:
            raise ValueError('Existing contract differs or --resume was not requested')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Refusing to overwrite a nonempty output directory without its contract')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(contract_path, contract)
    start = time.monotonic()
    atomic_json(output / 'data_manifest.json', {'state': 'running', 'contract': contract, 'argv': sys.argv,
                                                'completed_contexts': 0, 'planned_contexts': len(groups)})
    pending = []
    for group in groups:
        shard = output / 'shards' / f"{group[0]['context_id']}.json"
        if args.resume and shard.is_file():
            previous = json.loads(shard.read_text())
            if previous['context']['context_id'] != group[0]['context_id']:
                raise ValueError('A resumed shard has a different context')
            if previous['context']['status'] == 'complete':
                continue
        pending.append(group)
    # Spawned workers inherit these before importing NumPy. Each uses a single
    # numerical-library thread, making the worker count the parallelism limit.
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[variable] = '1'
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(work, group, args.max_depth, args.max_nodes, args.seed): group for group in pending}
        for future in concurrent.futures.as_completed(futures):
            group = futures[future]
            result = future.result()
            atomic_json(output / 'shards' / f"{group[0]['context_id']}.json", result)
    contexts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for group in groups:
        shard = output / 'shards' / f"{group[0]['context_id']}.json"
        if not shard.exists():
            contexts.append({**{key: group[0][key] for key in CONTEXT_FIELDS if key != 'source'},
                             'status': 'missing', 'error': 'context shard missing'})
            continue
        result = json.loads(shard.read_text())
        contexts.append(result['context'])
        candidates.extend(result['candidates'])
    atomic_jsonl(output / 'contexts.jsonl', contexts)
    atomic_jsonl(output / 'candidates.jsonl', candidates)
    states = Counter(row['status'] for row in contexts)
    variants = Counter(row.get('pool_variant') or 'unavailable' for row in contexts)
    manifest = {'state': 'complete' if states == {'complete': len(groups)} else 'partial', 'contract': contract,
                'planned_contexts': len(groups), 'completed_contexts': states['complete'],
                'status_counts': dict(states), 'pool_variants': dict(variants),
                'candidate_count': len(candidates), 'workers': args.workers, 'threads_per_worker': 1,
                'argv': sys.argv, 'elapsed_seconds': time.monotonic() - start,
                'output_sha256': {name: file_sha(output / name) for name in ('contexts.jsonl', 'candidates.jsonl')}}
    atomic_json(output / 'data_manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'evaluation'), required=True)
    parser.add_argument('--max-boards', type=int)
    parser.add_argument('--max-contexts', type=int, default=1)
    parser.add_argument('--max-depth', type=int, default=32)
    parser.add_argument('--max-nodes', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == '__main__':
    main()
