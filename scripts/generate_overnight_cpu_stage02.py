"""Generate full-board candidates from current states with a frozen local BART.

This stage never passes a demonstration target, action, offset or measurement
to the generator. Invalid and duplicate outputs remain in the raw records.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from transformers import BartForConditionalGeneration

from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.environment.training_goal import TrainingGoal
from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.boards import validate_subgoal
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'rl-data/validation/sokoban/components/full_data/generator/border/8/checkpoint-75856'
FAMILIES = ('12x12x4', '12x12x10')


def select_contexts(rows: list[dict[str, Any]], split: str, max_boards: int) -> list[dict[str, Any]]:
    """Select the frozen rank-zero contexts without exposing any later state."""
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (row.get('split') != split or row.get('context_rank') != 0
                or row.get('family') not in FAMILIES or row.get('board_rank', max_boards) >= max_boards):
            continue
        visible = {key: row[key] for key in ('context_id', 'layout_id', 'family', 'source_family', 'split',
                                             'board_rank', 'context_rank', 'source', 'source_hash')}
        previous = selected.setdefault(visible['context_id'], visible)
        if previous != visible:
            raise ValueError('Conflicting current-state identity in source data')
    family_order = {family: i for i, family in enumerate(FAMILIES)}
    result = sorted(selected.values(), key=lambda row: (row['board_rank'], family_order[row['family']]))
    if not result or any(not any(row['family'] == family for row in result) for family in FAMILIES):
        raise ValueError('Both families require a selected context')
    if len({(row['family'], row['layout_id']) for row in result}) != len(result):
        raise ValueError('Expected one context per canonical layout')
    for row in result:
        board = from_rows(row['source'])
        if state_hash(board) != row['source_hash'] or canonical_layout_id(board) != row['layout_id']:
            raise ValueError('Current board identity does not match the frozen data')
    return result


def decode_output(tokenizer: SokobanTokenizer, source: np.ndarray, tokens: list[int], rank: int) -> dict[str, Any]:
    """Keep the raw sequence and every decoded board, even when invalid."""
    target = tokenizer.board_detokenizer(tokens)
    reason = None
    try:
        if target is None:
            raise ValueError('could not decode a complete 12x12 board')
        validate_subgoal(source, target)
        if np.array_equal(source, target):
            raise ValueError('already satisfied')
    except (ValueError, AssertionError, TypeError) as error:
        reason = str(error)
    return {'rank': rank, 'raw_token_ids': tokens, 'valid_nontrivial': reason is None,
            'validation_error': reason, 'target': to_rows(target) if target is not None else None,
            'target_hash': state_hash(target) if target is not None else None}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_boards < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError('Positive board, batch and thread counts are required')
    started = time.monotonic()
    rows = read_jsonl(args.pairs)
    contexts = select_contexts(rows, args.split, args.max_boards)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = {'schema_version': 1, 'split': args.split, 'max_boards': args.max_boards,
                'pairs_path': str(args.pairs.resolve()), 'pairs_sha256': file_sha(args.pairs),
                'source_sha256': file_sha(Path(__file__)), 'checkpoint': str(MODEL),
                'weights_sha256': file_sha(MODEL / 'pytorch_model.bin'),
                'config_sha256': file_sha(MODEL / 'config.json'),
                'generation_config_sha256': file_sha(MODEL / 'generation_config.json'),
                'input_contract': 'current board only', 'device': 'cpu', 'dtype': 'float32',
                'torch': torch.__version__, 'batch_size': args.batch_size, 'threads': args.threads,
                'num_beams': 4, 'num_return_sequences': 2, 'max_new_tokens': 145,
                'do_sample': False, 'seed': 20261115}
    contract_path = output / 'contract.json'
    if contract_path.exists():
        if not args.resume or json.loads(contract_path.read_text()) != contract:
            raise ValueError('Existing generator contract differs or --resume was not requested')
    else:
        atomic_json(contract_path, contract)
    contract_hash = digest(contract)
    plan = {'schema_version': 1, 'contract_sha256': contract_hash,
            'contexts': [{key: row[key] for key in ('context_id', 'layout_id', 'family', 'split',
                                                     'board_rank', 'source_hash')} for row in contexts]}
    plan_path = output / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Existing generator context plan differs')
    else:
        atomic_json(plan_path, plan)
    shards = output / 'shards'
    shards.mkdir(exist_ok=True)
    pending = []
    for row in contexts:
        path = shards / f"{row['context_id']}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if (not args.resume or saved.get('contract_sha256') != contract_hash
                    or saved.get('context_id') != row['context_id'] or saved.get('source_hash') != row['source_hash']):
                raise ValueError('Existing generator shard does not match the frozen contract')
        else:
            pending.append(row)
    initial_swap = psutil.swap_memory().used
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261115)
    tokenizer = SokobanTokenizer()
    model = BartForConditionalGeneration.from_pretrained(MODEL, local_files_only=True).eval()
    model.requires_grad_(False)
    reason = None
    peak_memory = psutil.virtual_memory().percent / 100
    for start in range(0, len(pending), args.batch_size):
        if time.time() >= args.stop_epoch or time.monotonic() >= args.stop_monotonic:
            reason = 'reporting reserve reached'
            break
        memory = psutil.virtual_memory().percent / 100
        peak_memory = max(peak_memory, memory)
        if memory >= .8 or psutil.swap_memory().used > initial_swap + 512 * 1024 * 1024:
            reason = 'host memory or swap guard'
            break
        batch = pending[start:start + args.batch_size]
        boards = [from_rows(row['source']) for row in batch]
        tokens = torch.cat([tokenizer.x_y_tokenizer(board, board, TrainingGoal.GENERATOR)[0] for board in boards])
        with torch.inference_mode():
            generated = model.generate(tokens, max_new_tokens=145, num_beams=4,
                                       num_return_sequences=2, do_sample=False).tolist()
        if len(generated) != 2 * len(batch):
            raise ValueError('Generator returned an unexpected number of sequences')
        for index, row in enumerate(batch):
            outputs = [decode_output(tokenizer, boards[index], generated[index * 2 + rank], rank)
                       for rank in range(2)]
            atomic_json(shards / f"{row['context_id']}.json",
                        {'schema_version': 1, 'contract_sha256': contract_hash,
                         **{key: row[key] for key in ('context_id', 'layout_id', 'family', 'source_family',
                                                      'split', 'board_rank', 'source_hash')}, 'outputs': outputs})
        completed = len(contexts) - len(pending) + min(start + args.batch_size, len(pending))
        atomic_json(output / 'status.json', {'state': 'running', 'completed_contexts': completed,
                                            'planned_contexts': len(contexts), 'stop_reason': None,
                                            'elapsed_seconds': time.monotonic() - started,
                                            'peak_host_memory_fraction': peak_memory})
    count = sum((shards / f"{row['context_id']}.json").is_file() for row in contexts)
    result = {'state': 'complete' if count == len(contexts) else 'partial', 'completed_contexts': count,
              'planned_contexts': len(contexts), 'stop_reason': reason, 'elapsed_seconds': time.monotonic() - started,
              'peak_host_memory_fraction': peak_memory, 'swap_growth_bytes': max(0, psutil.swap_memory().used - initial_swap)}
    atomic_json(output / 'status.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'evaluation'), required=True)
    parser.add_argument('--max-boards', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--stop-epoch', type=float, required=True)
    parser.add_argument('--stop-monotonic', type=float, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
