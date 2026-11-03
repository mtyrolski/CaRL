"""Freeze random historical AdaSubS boards and replay their genuine saved solutions."""
import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from carl.icl.boards import canonical_layout_id, state_hash, to_rows
from carl.icl.data_sources.sokoban import verify_solution
from carl.planners.base import Experience
from scripts.trained_reference.common import ROOT, digest, save, selected_indices
from scripts.trained_reference.report import write_report


def matching_evidence(checkout: Path) -> dict[str, Any]:
    source = checkout / 'analysis/proposal_plots/matched_config_evidence.json'
    rows = [row for row in json.loads(source.read_text()) if row['tag'] == 'k841_12x12x4']
    expected = {'algorithm/solver/max_nodes': 500,
                'algorithm/solver/subgoal_generator/generator_k_list': '[8, 4, 1]',
                'algorithm/solver/subgoal_generator/subgoal_generation_kwargs/do_sample': False,
                'algorithm/solver/subgoal_generator/subgoal_generation_kwargs/num_beams': 4,
                'algorithm/solver/subgoal_generator/subgoal_generation_kwargs/num_return_sequences': 1}
    if not rows or any(any(row['parameters'].get(key) != value for key, value in expected.items()) for row in rows):
        raise ValueError('Historical k841 configuration evidence does not match requested reference')
    raw_logs = {row['source']: digest(checkout / row['source']) for row in rows
                if (checkout / row['source']).is_file()}
    return {'source': str(source), 'source_sha256': digest(source), 'matched_logs': len(rows),
            'available_raw_log_sha256': raw_logs, 'configurations': rows,
            'limitation': 'Exact tags match; batch files do not carry a run UUID or configuration hash.'}


def checkpoint_inventory(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config['checkpoint_root'])
    result = {}
    for role, relative in config['checkpoint_paths'].items():
        directory = root / relative
        result[role] = {'path': str(directory), 'current_file_sha256': {
            name: digest(directory / name) for name in ('config.json', 'pytorch_model.bin')},
            'historical_weight_identity_verified': False}
    return result


def require_same_or_save(path: Path, value: dict[str, Any]) -> None:
    canonical = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != canonical:
        raise ValueError(f'Frozen artifact differs; use a separate output directory: {path}')
    save(path, value)


def export_record(record: Experience, source: Path, source_hash: str, index: int,
                  board: dict[str, Any], cap: int) -> tuple[dict[str, Any], dict[str, Any]]:
    info, solution = record.search_info, record.solution
    low_level = info.low_level_nodes_visited
    if low_level is None or low_level < 0 or info.high_level_nodes_valid is None:
        raise ValueError(f'Missing work counters in {source}:{index}')
    selected = info.subgoals_selected_for_expansion
    if set(selected) != {8, 4, 1}:
        raise ValueError(f'Unexpected generator bank: {source}:{index}')
    replayed = None
    actions = solution.action_path
    if solution.solved:
        if actions is None or not solution.subgoal_path:
            raise ValueError('Solved record lacks action/subgoal path')
        initial = info.search_tree.state if info.search_tree is not None else solution.subgoal_path[0]
        if not isinstance(initial, np.ndarray):
            raise ValueError('Expected a Sokoban array for solution replay')
        verify_solution(initial, actions)
        replayed = True
    elif actions is not None:
        raise ValueError('Unsolved record unexpectedly has a solution path')
    row = {'board_id': board['board_id'], 'state_hash': board['state_hash'],
           'population_index': board['population_index'], 'method': 'historical_adasubs_k841',
           'outcome_origin': 'saved_original_solver_record', 'solved': bool(solution.solved),
           'finish_reason': info.finished_reason, 'native_low_level_cap': cap,
           'low_level_nodes_visited': low_level, 'native_overshoot': max(0, low_level - cap),
           'generator_calls': sum(selected.values()), 'value_calls': info.high_level_nodes_valid,
           'accepted_subgoals': info.high_level_nodes_valid,
           'rejected_subgoals': info.high_level_nodes_unreachable,
           'solution_length': len(actions) if actions is not None else None,
           'solution_replayed': replayed, 'runtime_seconds': getattr(info, 'runtime_seconds', None),
           'historical_search_seed': None, 'source_batch': str(source), 'source_sha256': source_hash,
           'source_index': index}
    detail = {**row, 'initial_state': board['state'], 'action_path': actions,
              'subgoal_path': [to_rows(state) for state in solution.subgoal_path] if solution.subgoal_path else None,
              'subgoal_distance_path': solution.subgoal_distance_path,
              'subgoals_selected_for_expansion': selected,
              'subgoals_reachable_count_per_k': info.subgoals_reachable_count_per_k,
              'subgoals_unreachable_count_per_k': info.subgoals_unreachable_count_per_k}
    return row, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/trained_reference/random100_k841.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    checkout = Path(config['source_checkout'])
    source_boards = Path(config['source_boards'])
    output = ROOT / config['output']
    population_size = config['population_size']
    chosen = selected_indices(population_size, config['sample_size'], config['selection_seed'])
    boards = joblib.load(source_boards)
    if len(boards) < population_size or any(state.shape != (12, 12, 7) for state in boards[:population_size]):
        raise ValueError('Source population is not the expected full 12x12 state collection')
    population_hashes = [state_hash(state) for state in boards[:population_size]]
    if len(set(population_hashes)) != population_size:
        raise ValueError('Duplicate initial states in the sampling population')
    public = [{'board_id': f'real_12x12x4-{population_hashes[i][:24]}',
               'state_hash': population_hashes[i], 'layout_id': canonical_layout_id(boards[i]),
               'family': 'real_12x12x4', 'split': 'historical_reference100',
               'population_index': i, 'state': to_rows(boards[i])} for i in chosen]
    selection = {'schema_version': 1, 'selection_seed': config['selection_seed'],
                 'selection_algorithm': 'NumPy Generator(PCG64(seed)).choice(N,n,replace=False), sorted indices',
                 'population_size': population_size, 'sample_size': len(chosen),
                 'source_boards': str(source_boards), 'source_boards_sha256': digest(source_boards),
                 'outcomes_used_for_selection': False, 'instances': public}
    # Write the board cohort before deserializing solver outcomes.
    require_same_or_save(output / 'boards.json', selection)
    sources = sorted((checkout / 'solve_attempts').glob('solved_problems_k841_12x12x4_batch_*.joblib'),
                     key=lambda path: int(path.stem.rsplit('_', 1)[1]))
    if len(sources) != 20:
        raise ValueError('Expected all 20 historical k841 batches, including failures')
    evidence = matching_evidence(checkout)
    freeze = {'schema_version': 1, 'status': 'frozen', 'config': config,
              'boards_sha256': digest(output / 'boards.json'),
              'source_batch_sha256': {str(path): digest(path) for path in sources},
              'checkpoint_inventory': checkpoint_inventory(config),
              'config_evidence_sha256': evidence['source_sha256'],
              'native_counter_semantics': 'Executed validator steps; checked before expansion, overshoot possible.',
              'historical_training_overlap': 'unknown', 'new_inference_required': False}
    require_same_or_save(output / 'freeze.json', freeze)
    save(output / 'config_evidence.json', evidence)
    selected_boards = {row['population_index']: row for row in public}
    rows = []
    total, total_solved = 0, 0
    for source in sources:
        batch = joblib.load(source)
        if len(batch) != 50:
            raise ValueError(f'Expected 50 attempts in {source}')
        for index, record in enumerate(batch):
            if not isinstance(record, Experience) or record.search_info.search_tree is None:
                raise ValueError(f'Missing original attempt/root in {source}:{index}')
            if state_hash(record.search_info.search_tree.state) != population_hashes[total]:
                raise ValueError(f'Historical attempt does not match ordered population at {total}')
            if total in selected_boards:
                row, detail = export_record(record, source, freeze['source_batch_sha256'][str(source)],
                                            index, selected_boards[total], config['native_low_level_cap'])
                rows.append(row)
                require_same_or_save(output / 'records' / f"{row['board_id']}.json", detail)
            total_solved += bool(record.solution.solved)
            total += 1
        print(f'Checked {total}/{population_size} archive records; exported {len(rows)}/{len(chosen)}', flush=True)
    if total != population_size or len(rows) != len(chosen):
        raise ValueError('Incomplete population or selection')
    pilot = json.loads((ROOT / 'data/icl/campaign_dataset_12x12_v1/real_12x12x4/pilot100.json').read_text())
    overlap = set(population_hashes) & {row['state_hash'] for row in pilot['instances']}
    audit = {'records': total, 'solved': total_solved, 'unsolved': total - total_solved,
             'unique_initial_states': len(set(population_hashes)), 'source_order_verified': True,
             'failures_retained_in_population': True, 'current_pilot_board_overlap': len(overlap),
             'selected_unique_layouts': len({row['layout_id'] for row in public})}
    write_report(output, rows, audit)
    save(output / 'artifact_hashes.json', {str(path.relative_to(output)): digest(path)
         for path in sorted(output.rglob('*')) if path.is_file() and path.name not in {'artifact_hashes.json', 'reproducibility.json'}})
    print(json.dumps(json.loads((output / 'summary.json').read_text()), indent=2))


if __name__ == '__main__':
    main()
