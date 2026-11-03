"""Prepare fresh, replay-verified validation contexts without opening test boards."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows, tiles
from carl.icl.campaign_data import public_board, sha256_file
from carl.icl.datasets import verify_solution

ROOT = Path('data/icl/campaign_dataset_v2')
OUT = Path('data/icl/campaign_policy_validation_v1')
PRIVATE = Path('data/icl/campaign_policy_validation_v1_private')
OFFSETS = (1, 2, 4, 8)


def distance(board: np.ndarray, start: tuple[int, int], end: tuple[int, int], *, boxes: bool) -> int | None:
    """Shortest walk with walls and, optionally, stationary boxes as obstacles."""
    blocked = {0, 3, 4} if boxes else {0}
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), count = queue.popleft()
        if (r, c) == end:
            return count
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            point = (r + dr, c + dc)
            if (0 <= point[0] < board.shape[0] and 0 <= point[1] < board.shape[1]
                    and point not in seen and board[point] not in blocked):
                seen.add(point)
                queue.append((point, count + 1))
    return None


def contexts(states: list[np.ndarray]) -> list[dict]:
    """Find matched offsets and classify the witness without exposing its moves."""
    result = []
    cells = [tiles(state) for state in states]
    for i in range(len(states) - 8):
        if any(np.array_equal(states[i], states[i + k]) for k in OFFSETS):
            continue
        # Avoid duplicate targets for different offsets on a looping witness.
        if len({states[i + k].tobytes() for k in OFFSETS}) != len(OFFSETS):
            continue
        pushes = [not np.array_equal(np.isin(cells[j], [3, 4]), np.isin(cells[j + 1], [3, 4]))
                  for j in range(i, i + 8)]
        labels = {'walking'} if not all(pushes) else set()
        if any(pushes):
            labels.add('pushing')
        for k in OFFSETS:
            if any(pushes[:k]):
                continue
            player_a = np.argwhere(np.isin(cells[i], [5, 6]))[0]
            player_b = np.argwhere(np.isin(cells[i + k], [5, 6]))[0]
            a = (int(player_a[0]), int(player_a[1]))
            b = (int(player_b[0]), int(player_b[1]))
            direct = distance(cells[i], a, b, boxes=False)
            around = distance(cells[i], a, b, boxes=True)
            if direct is not None and around is not None and direct < around <= k:
                labels.add('stationary_box_detour')
        result.append({'position': i, 'labels': sorted(labels)})
    return result


def main() -> None:
    history = json.loads(Path('analysis/luna_icl/campaign_100/history_snapshot.json').read_text())
    excluded = set(history['observed_layout_ids']) | set(history['reserved_layout_ids'])
    # Read only metadata, never board states or witnesses from the pilot/holdout.
    for path in ROOT.glob('*/*.json'):
        data = json.loads(path.read_text())
        excluded.update(item['layout_id'] for item in data.get('instances', []))
    training = json.loads((ROOT / 'training.json').read_text())
    excluded.update(canonical_layout_id(from_rows(item['state'])) for item in training['contexts'])
    candidates = []
    for path in Path('data/icl/.private_campaign_real_staging').glob('*.json'):
        row = json.loads(path.read_text())
        if row['layout_id'] in excluded:
            continue
        states = verify_solution(from_rows(row['state']), row['actions'])
        options = contexts(states)
        if options:
            candidates.append((row['rank'], row, options))
    candidates.sort(key=lambda item: item[0])
    chosen = []
    # First require an actual stationary-box detour, then a pushing segment.
    for label in ('stationary_box_detour', 'pushing', 'walking', 'walking'):
        for _, row, options in candidates:
            if row['layout_id'] in {item[0]['layout_id'] for item in chosen}:
                continue
            matching = [item for item in options if label in item['labels']]
            if matching:
                chosen.append((row, matching[0]))
                break
        else:
            raise RuntimeError(f'Fresh validation lacks category: {label}')
    records, witnesses, selections = [], [], []
    for row, choice in chosen:
        record = public_board(rows=row['state'], family='real_12x12x4', split='component_validation',
                              trajectory_id=row['candidate_id'], source_path=row['source_path'],
                              source_sha256=row['source_sha256'], source_index=row['source_index'],
                              generator_version=None, seed=None)
        records.append(record)
        witnesses.append({'board_id': record['id'], 'actions': row['actions']})
        selections.append({'board_id': record['id'], 'layout_id': row['layout_id'], **choice})
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    for file in ('training.json', 'value_examples.json'):
        (OUT / file).write_bytes((ROOT / file).read_bytes())
    manifest_path = OUT / 'real_12x12x4/component_validation.json'
    write_json(manifest_path, {'provenance': {'source': 'verified fresh staging; selection policy v1'}, 'instances': records})
    write_json(OUT / 'campaign_dataset_manifest.json', {'version': 1, 'manifest_sha256': {
        'real_12x12x4/component_validation.json': sha256_file(manifest_path)}})
    write_json(PRIVATE / 'component_validation/real_12x12x4.json', {'version': 2, 'witnesses': witnesses})
    write_json(OUT / 'selection.json', {'version': 1, 'offsets': OFFSETS, 'selections': selections,
                                        'fresh_candidates': len(candidates), 'excluded_layout_count': len(excluded)})
    print(json.dumps({'fresh_candidates': len(candidates), 'boards': len(records),
                      'contexts': [item['labels'] for item in selections],
                      'public_manifest_sha256': sha256_file(manifest_path)}, indent=2))


if __name__ == '__main__':
    main()
