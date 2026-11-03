"""Compare saved Torch and Luna validation probes on identical full states."""
from typing import Any

from carl.icl.boards import from_rows, state_hash
from carl.icl.data_sources.sokoban import verify_solution
from scripts.plot_data_export.common import CAMPAIGN, DATA, ROOT, read_json, sha


def comparison() -> dict[str, list[dict[str, Any]]]:
    manifest = read_json(DATA / 'real_12x12x4/scale_validation.json')
    boards = {row['board_id']: row for row in manifest['instances']}
    freeze = read_json(CAMPAIGN / 'scale_tuning/real12_v1/freeze.json')
    rows = []
    for path in sorted((ROOT / 'analysis/torch_checkpoint_probe').glob('12x12_validation_board*_b128.json')):
        record = read_json(path)
        board = boards[record['board']['id']]
        outcome = record['outcome']
        initial = from_rows(outcome['subgoal_path'][0])
        if state_hash(initial) != board['state_hash'] or record['board']['state_hash'] != board['state_hash']:
            raise ValueError(f'Torch state does not match validation manifest: {path}')
        if not record['verification']['verified'] or record['config']['action_budget'] != 128:
            raise ValueError(f'Unverified or unmatched Torch budget: {path}')
        if outcome['solved']:
            verify_solution(initial, outcome['action_path'])
        rows.append({'board_id': board['board_id'], 'state_hash': board['state_hash'],
                     'method': 'torch_fixed_k', 'model': 'pretrained BART/BERT',
                     'scales': record['config']['generator_scales'], 'budget_actions': 128,
                     'seed': record['config']['search_seed'], 'solved': outcome['solved'],
                     'replay_verified': record['verification']['verified'],
                     'attempted_actions': outcome['attempted_actions'],
                     'solution_length': len(outcome['action_path']) if outcome['solved'] else None,
                     'expansions': outcome['expansions'], 'finish_reason': outcome['finish_reason'],
                     'policy_mode': 'greedy argmax', 'source': str(path.relative_to(ROOT)), 'source_sha256': sha(path)})
    if len(rows) != 2 or len({row['board_id'] for row in rows}) != 2:
        raise ValueError('Expected exactly two distinct saved Torch validation probes')
    selected_boards = {row['board_id'] for row in rows}
    for method in ('relative_u', 'absolute_d'):
        scales = freeze['conditions'][method]['config']['scales']
        for path in sorted((CAMPAIGN / 'scale_tuning/real12_v1/solver').glob(f'tune_{method}_*.json')):
            record = read_json(path)
            if record['config']['scales'] != scales:
                continue
            outcome = record['outcome']
            board = boards[record['instance']['id']]
            if state_hash(from_rows(record['instance']['state'])) != board['state_hash']:
                raise ValueError(f'Luna state does not match validation manifest: {path}')
            if not record['verification']['verified'] or record['config']['action_budget'] != 128:
                raise ValueError(f'Unverified or unmatched Luna budget: {path}')
            rows.append({'board_id': board['board_id'], 'state_hash': board['state_hash'],
                         'method': method, 'model': record['config']['model'], 'scales': scales,
                         'budget_actions': 128, 'seed': record['seed'], 'solved': outcome['solved'],
                         'replay_verified': record['verification']['verified'],
                         'attempted_actions': outcome['attempted_actions'],
                         'solution_length': len(outcome['action_path']) if outcome['solved'] else None,
                         'expansions': outcome['expansions'], 'finish_reason': outcome['finish_reason'],
                         'policy_mode': 'sampled categorical', 'source': str(path.relative_to(ROOT)),
                         'source_sha256': sha(path)})
    summary = []
    for method in ('torch_fixed_k', 'relative_u', 'absolute_d'):
        group = [row for row in rows if row['method'] == method]
        if len(group) != 2 or {row['board_id'] for row in group} != selected_boards:
            raise ValueError(f'Unmatched validation board pairs: {method}')
        solved = sum(row['solved'] for row in group)
        summary.append({'method': method, 'budget_actions': 128, 'boards': len(group),
                        'solved': solved, 'solve_fraction': solved / len(group),
                        'total_actions': sum(row['attempted_actions'] for row in group),
                        'policy_mode': group[0]['policy_mode'], 'split': 'scale_validation'})
    return {'torch_luna_validation_runs': rows, 'torch_luna_validation_summary': summary}
