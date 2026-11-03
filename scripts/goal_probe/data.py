"""Freeze small, TRAIN-only goal-placement demonstrations and targeted validation."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import joblib
import numpy as np

from carl.icl.boards import canonical_layout_id, from_rows, state_hash, tiles, to_rows
from carl.icl.config import ROOT
from carl.icl.datasets import environment
from carl.icl.types import Board
from scripts.generator_probe.prepare_dataset import write_once
from scripts.generator_probe.prepare_sources import sha256
from scripts.datasets.prepare_fresh_trajectory_holdout import verified_trajectory

SOURCE = ROOT / 'analysis/luna_icl/campaign_100/generator_probe_v2'
OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/goal_progress_probe'
LENGTHS = (1, 2, 4, 8, 3, 6)


class Origin(TypedDict):
    """Existing permitted source identity, independent of its continuation."""
    board_id: str
    trajectory_id: str
    source_path: str
    source_index: int
    source_sha256: str
    layout_id: str
    position: int
    state_hash: str


class Query(TypedDict):
    """Public validation source; no witness or target identity."""
    id: str
    state: list[str]
    provenance: Origin


class Demonstration(Query):
    """Mechanics evidence, explicitly without calibrated scale labels."""
    subgoal: list[str]
    actions: list[int]
    k: int
    goals_before: int
    goals_after: int
    target_position: int
    target_state_hash: str
    walking_actions: int
    pushing_actions: int


class OriginalRecord(TypedDict):
    """Fields consumed from the existing train/validation manifests."""
    id: str
    board_id: str
    trajectory_id: str
    source_path: str
    source_index: int
    source_sha256: str
    layout_id: str
    state_hash: str
    state: list[str]


@dataclass
class Verified:
    """Only an explicitly allowed original trajectory is inspected."""
    record: OriginalRecord
    states: list[Board]
    actions: list[int]


def goals(rows: list[str]) -> int:
    """Count occupied goals, including no player-on-goal tiles."""
    return sum(row.count('*') for row in rows)


def choose_window(source: Verified, length: int) -> tuple[int, int]:
    """Earliest increasing-goal push with the requested exact preceding window."""
    counts = [int(np.count_nonzero(tiles(state) == 3)) for state in source.states]
    for end in range(length, len(source.states)):
        start = end - length
        if counts[end] > counts[end - 1] and counts[end] > counts[start]:
            return start, end
    raise ValueError(f'No goal-placement window: {source.record["id"]}, {length}')


def snippet(source: Verified, length: int, split: str) -> Demonstration:
    """Export recorded legal actions and their exact replayed endpoint."""
    start, end = choose_window(source, length)
    record = source.record
    before, after = source.states[start], source.states[end]
    state, target = to_rows(before), to_rows(after)
    actions = source.actions[start:end]
    env = environment(before)
    current = before
    pushes = 0
    for action, expected in zip(actions, source.states[start + 1:end + 1], strict=True):
        successor = cast(Board, env.next_state(current, action))
        if not np.array_equal(successor, expected):
            raise ValueError('Selected snippet replay differs from original')
        pushes += int(not np.array_equal(np.isin(tiles(current), [3, 4]), np.isin(tiles(successor), [3, 4])))
        current = successor
    origin = Origin(board_id=record['board_id'], trajectory_id=record['trajectory_id'],
                    source_path=record['source_path'], source_index=record['source_index'],
                    source_sha256=record['source_sha256'], layout_id=record['layout_id'],
                    position=start, state_hash=state_hash(before))
    return Demonstration(id=f'{split}:{record["trajectory_id"]}:{start}:{end}', state=state,
                         provenance=origin, subgoal=target, actions=actions, k=length,
                         goals_before=goals(state), goals_after=goals(target), target_position=end,
                         target_state_hash=state_hash(after), walking_actions=length - pushes, pushing_actions=pushes)


def load_sources(records: list[OriginalRecord], trajectories: list[Board]) -> list[Verified]:
    """Replay only the six selected allowed trajectories, never protected boards."""
    result = []
    for record in records[:6]:
        states, actions = verified_trajectory(np.asarray(trajectories[record['source_index']]))
        if canonical_layout_id(states[0]) != record['layout_id']:
            raise ValueError('Original layout identity differs')
        if state_hash(states[0]) != record['state_hash'] or to_rows(states[0]) != record['state']:
            raise ValueError('Original trajectory initial state differs')
        result.append(Verified(record, states, actions))
    return result


def baseline_evidence() -> dict[str, object]:
    """Replay historical validation tree edges and count every reached intermediate."""
    path = SOURCE / 'solver_validation/case.json'
    case = json.loads(path.read_text())
    nodes = case['tree']['nodes']
    states: dict[int, Board] = {0: from_rows(nodes[0]['state'])}
    counts = [goals(nodes[0]['state'])]
    action_count = 0
    for node in nodes[1:]:
        current = states[node['parent_id']]
        env = environment(current)
        for action in node['actions']:
            current = cast(Board, env.next_state(current, action))
            counts.append(goals(to_rows(current)))
            action_count += 1
        if to_rows(current) != node['state']:
            raise ValueError('Historical baseline tree edge replay mismatch')
        states[node['node_id']] = current
    if action_count != case['outcome']['attempted_actions']:
        raise ValueError('Historical baseline action accounting differs')
    return {'source_path': str(path.relative_to(ROOT)), 'source_sha256': sha256(path),
            'historical_validation_only': True, 'new_model_calls': 0,
            'replayed_actions': action_count, 'initial_occupied_goals': counts[0],
            'maximum_occupied_goals_over_all_replayed_states': max(counts),
            'states_with_occupied_goals': sum(count > 0 for count in counts),
            'solved': case['outcome']['solved'], 'finish_reason': case['outcome']['finish_reason']}


def prepare(output: Path = OUTPUT) -> None:
    """Prepare immutable artifacts offline; witness selection is disclosed explicitly."""
    training_path = SOURCE / 'training_candidates.json'
    validation_path = SOURCE / 'validation_sources_30.json'
    checks = json.loads((SOURCE / 'dataset_provenance_checks.json').read_text())
    if sha256(training_path) != checks['training_manifest_sha256']:
        raise ValueError('Training manifest differs from verified provenance')
    if sha256(validation_path) != checks['validation_manifest_sha256']:
        raise ValueError('Validation manifest differs from verified provenance')
    train = cast(list[OriginalRecord], json.loads(training_path.read_text())['contexts'])
    rows = cast(list[OriginalRecord], json.loads(validation_path.read_text())['instances'])
    validation = list({row['layout_id']: row for row in reversed(rows)}.values())
    validation.reverse()  # Original layout order; keep position zero from each layout.
    train_ids = {row['layout_id'] for row in train}
    val_ids = {row['layout_id'] for row in validation}
    exclusion = SOURCE / 'dataset_selection_exclusions.json'
    protected = set(json.loads(exclusion.read_text())['protected_layout_ids'])
    if len(train_ids) != 10 or len(val_ids) != 10 or train_ids & val_ids or (train_ids | val_ids) & protected:
        raise ValueError('Allowed train/validation/protected identities overlap')
    source_path = ROOT / checks['source_path']
    digest = sha256(source_path)
    if digest != checks['source_sha256'] or any(row['source_sha256'] != digest for row in train + validation):
        raise ValueError('Offline source checksum differs')
    raw = joblib.load(source_path)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    training = [snippet(item, length, 'train') for item, length in
                zip(load_sources(train, trajectories), LENGTHS, strict=True)]
    witnesses = [snippet(item, length, 'targeted_validation') for item, length in
                 zip(load_sources(validation, trajectories), LENGTHS, strict=True)]
    queries = [Query(id=row['id'], state=row['state'], provenance=row['provenance']) for row in witnesses]
    output.mkdir(parents=True, exist_ok=True)
    write_once(output / 'data_training_demonstrations.json', {
        'version': 1, 'split': 'train_only', 'description': 'Verified task mechanics; no u/d labels or calibration claims.',
        'action_order': ['UP', 'DOWN', 'LEFT', 'RIGHT'], 'demonstrations': training})
    write_once(output / 'data_validation_queries.json', {
        'version': 1, 'split': 'targeted_goal_placement_validation',
        'description': 'Witness-selected near-goal component queries; not full-board solves or an unbiased validation sample.',
        'instances': queries})
    write_once(output / 'data_private_validation_witnesses.json', {
        'version': 1, 'private_evaluation_only': True, 'never_include_in_model_query': True, 'witnesses': witnesses})
    write_once(output / 'data_selection_checks.json', {
        'version': 1, 'selection_rule': 'First six layouts in existing manifest order; per-layout exact window lengths '
        '1,2,4,8,3,6; earliest recorded endpoint whose last action increases occupied goals and whose net gain is positive.',
        'witness_selection_bias': 'Targeted task-progress component validation only; not full-board solving/generalization.',
        'training_manifest_sha256': sha256(training_path), 'validation_manifest_sha256': sha256(validation_path),
        'protected_identity_snapshot_sha256': sha256(exclusion), 'source_sha256': digest,
        'train_layouts': 6, 'validation_layouts': 6, 'train_validation_intersections': 0,
        'protected_intersections': 0, 'new_model_calls': 0, 'all_selected_trajectories_and_snippets_replayed': True,
        'training_goals_before_after': [[row['goals_before'], row['goals_after']] for row in training],
        'training_walking_actions': [row['walking_actions'] for row in training],
        'policy_relative_labels': 'Absent; any semantic-goal calibration requires actual frozen-policy execution.',
        'baseline': baseline_evidence()})
    print(f'Verified six training demonstrations and six targeted validation queries: {output}')


if __name__ == '__main__':
    prepare()
