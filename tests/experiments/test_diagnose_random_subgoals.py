"""Random-walk targets and BFS results remain reproducible and auditable."""
import json

import numpy as np

from carl.icl.boards import from_rows, to_rows
from carl.icl.datasets import environment
from scripts.diagnose_random_subgoals import DEFAULT_SOURCE, audit_target, random_targets, run_jobs


def test_parallel_results_and_bounds() -> None:
    source = from_rows(['######', '#@---#', '#-$.-#', '#----#', '#----#', '######'])
    env = environment(source)
    target = np.asarray(env.next_state(source, 3))
    base = {'status': 'ready', 'source': to_rows(source), 'target': to_rows(target)}
    jobs = [{**base, 'depth_limit': 1, 'max_states': 100},
            {**base, 'depth_limit': 0, 'max_states': 100},
            {**base, 'depth_limit': 1, 'max_states': 1}]
    serial = run_jobs(jobs, 1)
    assert serial == run_jobs(jobs, 2)
    assert [row['status'] for row in serial] == ['reachable', 'unreachable_within_cap', 'node_limit']
    assert serial[0]['bfs_distance'] == 1 and serial[0]['witness_replayed']
    assert serial[1]['bfs_distance'] is None and serial[2]['bfs_distance'] is None


def test_random_walk_uses_exact_replay_and_seed() -> None:
    instance = json.loads(DEFAULT_SOURCE.read_text())['instances'][0]
    rows, board_id = instance['state'], instance['id']
    first = random_targets(rows, board_id, 17, [1, 2, 4, 8])
    assert first == random_targets(rows, board_id, 17, [1, 2, 4, 8])
    source = from_rows(rows)
    env = environment(source)
    for row in first:
        if row['status'] != 'ready':
            continue
        current = source.copy()
        for action in row['walk_actions']:
            assert not env.is_solved(current)
            current = np.asarray(env.next_state(current, action))
        assert to_rows(current) == row['target']
        result = audit_target({**row, 'depth_limit': 8, 'max_states': 2500})
        if result['status'] == 'reachable':
            assert result['witness_replayed'] and result['bfs_distance'] <= row['offset']
