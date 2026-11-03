"""Prepare 10 training layouts and 30 validation sources from offline rl-data.

Eight original training and four original validation layouts are reused. Eight
fresh disjoint layouts provide two training and six validation trajectories.
Training prefers k=1,2,4 and replaces unchanged/repeated endpoints by distinct
recorded targets within k<=8. No policy-relative labels are invented.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import joblib
import numpy as np

from carl.icl.boards import canonical_layout_id, state_hash, to_rows
from carl.icl.config import ROOT
from carl.icl.types import Board, Candidate, TrainingContext
from scripts.generator_probe.prepare_sources import OriginalSource, SourceState, sha256, verified_sources
from scripts.generator_probe.source_selection import PART1, exclusion_snapshot, fresh_candidates
from scripts.datasets.prepare_fresh_trajectory_holdout import verified_trajectory

OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/generator_probe_v2'
DEMONSTRATIONS = ROOT / 'data/icl/campaign_dataset_v2/demonstrations.json'
TRAINING = ROOT / 'data/icl/campaign_dataset_v2/training.json'
TARGET_SELECTION_RULE = (
    'Try preferred offsets 1,2,4 in that order; reject source-equal or already selected endpoints. '
    'Fill vacancies from offsets 3,5,6,7,8 in that order, applying the same exclusions. '
    'Keep exactly three distinct non-source targets; sort by actual recorded k. '
    'Preserve recorded actions and actual k; never assign d/u without policy execution calibration.'
)


class RejectedTarget(TypedDict):
    """An honest trajectory offset excluded because its endpoint cannot be a subgoal."""

    k: int
    reason: str


class TargetSelection(TypedDict):
    """Auditable deterministic endpoint selection on a training-only trajectory."""

    preferred_offsets: list[int]
    selected_offsets: list[int]
    rejected_offsets: list[RejectedTarget]


@dataclass
class VerifiedSource:
    """Private preparation-time trajectory; only selected pieces are exported."""

    original: OriginalSource
    states: list[Board]
    actions: list[int]


class TrainingSource(TrainingContext):
    """Training-only examples and provenance, without fabricated d/u labels."""

    board_id: str
    trajectory_id: str
    source_path: str
    source_index: int
    source_sha256: str
    state_hash: str
    layout_id: str
    target_selection: TargetSelection


class TrainingManifest(TypedDict):
    """Thirty verified action-offset candidates on ten training layouts."""

    version: int
    description: str
    original_training_path: str
    original_training_sha256: str
    exclusion_snapshot_sha256: str
    target_selection_rule: str
    contexts: list[TrainingSource]


class ValidationManifest(TypedDict):
    """Thirty source positions with no future targets or actions."""

    version: int
    split: str
    description: str
    count: int
    distinct_layouts: int
    positions: list[int]
    exclusion_snapshot_sha256: str
    instances: list[SourceState]


def write_once(path: Path, document: object) -> None:
    """Preserve artifacts and reject incompatible reruns."""
    if path.exists():
        if json.loads(path.read_text()) != document:
            raise ValueError(f'Existing artifact differs; refusing to overwrite: {path}')
        return
    with path.open('x') as file:
        file.write(json.dumps(document, indent=2) + '\n')


def original_training(trajectories: list[Board]) -> list[VerifiedSource]:
    """Replay eight training trajectories and check every existing training target."""
    records = cast(list[OriginalSource], json.loads(DEMONSTRATIONS.read_text())['instances'])
    contexts = cast(list[TrainingContext], json.loads(TRAINING.read_text())['contexts'])
    if len(records) != 8 or len(contexts) != 8:
        raise ValueError('Expected eight original training contexts')
    verified = []
    for record, context in zip(records, contexts, strict=True):
        if record['source_path'] != PART1 or record['source_sha256'] != sha256(ROOT / PART1):
            raise ValueError('Original training source lineage differs')
        states, actions = verified_trajectory(np.asarray(trajectories[record['source_index']]))
        if context['state'] != to_rows(states[0]) or record['state'] != context['state']:
            raise ValueError('Original training context differs from the offline trajectory')
        if state_hash(states[0]) != record['state_hash'] or canonical_layout_id(states[0]) != record['layout_id']:
            raise ValueError('Original training source hashes differ')
        for candidate in context['candidates']:
            k = candidate['k']
            if candidate['actions'] != actions[:k] or candidate['subgoal'] != to_rows(states[k]):
                raise ValueError('Existing training candidate differs from its offline source')
        record = OriginalSource(**{**record, 'id': context['id']})
        verified.append(VerifiedSource(record, states, actions))
    return verified


def source_record(index: int, states: list[Board], digest: str) -> OriginalSource:
    """Create public source lineage for an already verified fresh trajectory."""
    hashed = state_hash(states[0])
    board_id = f'real_12x12x4-{hashed[:24]}'
    return OriginalSource(id=board_id, board_id=board_id, trajectory_id=f'part_1:{index}',
                          source_path=PART1, source_index=index, source_sha256=digest,
                          state=to_rows(states[0]), state_hash=hashed,
                          layout_id=canonical_layout_id(states[0]))


def selected_targets(source: VerifiedSource) -> tuple[list[Candidate], TargetSelection]:
    """Keep three distinct non-source endpoints and the exact recorded action offsets."""
    initial = state_hash(source.states[0])
    seen: set[str] = set()
    accepted: list[int] = []
    rejected: list[RejectedTarget] = []
    for k in (1, 2, 4, 3, 5, 6, 7, 8):
        if k >= len(source.states):
            rejected.append(RejectedTarget(k=k, reason='offset_not_available'))
            continue
        target = state_hash(source.states[k])
        reason = 'source_equal' if target == initial else 'duplicate_selected_target' if target in seen else None
        if reason is not None:
            rejected.append(RejectedTarget(k=k, reason=reason))
            continue
        accepted.append(k)
        seen.add(target)
        if len(accepted) == 3:
            break
    if len(accepted) != 3:
        raise ValueError(f"Training source {source.original['id']} lacks three distinct non-source targets within k<=8")
    accepted.sort()
    candidates = [Candidate(k=k, subgoal=to_rows(source.states[k]), actions=source.actions[:k]) for k in accepted]
    selection = TargetSelection(preferred_offsets=[1, 2, 4], selected_offsets=accepted, rejected_offsets=rejected)
    return candidates, selection


def training_context(source: VerifiedSource) -> TrainingSource:
    """Expose three legal recorded training transitions, with no measured scale claim."""
    record = source.original
    candidates, selection = selected_targets(source)
    return TrainingSource(
        id=record['id'], state=record['state'], remaining_actions=len(source.actions),
        candidates=candidates, calibration_indices=[0, 1, 2],
        board_id=record['board_id'], trajectory_id=record['trajectory_id'],
        source_path=record['source_path'], source_index=record['source_index'],
        source_sha256=record['source_sha256'], state_hash=record['state_hash'], layout_id=record['layout_id'],
        target_selection=selection,
    )


def validation_sources(sources: list[VerifiedSource]) -> list[SourceState]:
    """Expose three source positions per layout, omitting all continuation information."""
    instances = []
    for source in sources:
        record = source.original
        for position in (0, 1, 2):
            state = source.states[position]
            instances.append(SourceState(
                id=f"{record['id']}:position_{position:02d}", board_id=record['board_id'],
                trajectory_id=record['trajectory_id'], source_path=record['source_path'],
                source_index=record['source_index'], source_sha256=record['source_sha256'],
                position=position, state=to_rows(state), state_hash=state_hash(state), layout_id=record['layout_id'],
            ))
    if len(instances) != 30 or len({row['state_hash'] for row in instances}) != 30:
        raise ValueError('Expected thirty distinct validation source states')
    return instances


def prepare(output: Path) -> None:
    """Freeze selection exclusions and build the matched training/validation dataset."""
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / 'dataset_selection_exclusions.json'
    snapshot = exclusion_snapshot(snapshot_path)
    raw = joblib.load(ROOT / PART1)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    training = original_training(trajectories)
    validation = []
    for record, states in verified_sources():
        replayed, actions = verified_trajectory(np.asarray(trajectories[record['source_index']]))
        if any(not np.array_equal(a, b) for a, b in zip(states, replayed, strict=True)):
            raise ValueError('Original validation replay differs')
        validation.append(VerifiedSource(record, states, actions))
    fresh = []
    rejected: list[int] = []
    for candidate in fresh_candidates(snapshot):
        try:
            states, actions = verified_trajectory(np.asarray(trajectories[candidate['source_index']]))
        except (ValueError, IndexError, TypeError):
            rejected.append(candidate['source_index'])
            continue
        if (len(actions) < 16 or state_hash(states[0]) != candidate['state_hash']
                or canonical_layout_id(states[0]) != candidate['layout_id']):
            rejected.append(candidate['source_index'])
            continue
        record = source_record(candidate['source_index'], states, snapshot['source_sha256'])
        fresh.append(VerifiedSource(record, states, actions))
        if len(fresh) == 8:
            break
    if len(fresh) != 8:
        raise ValueError('Fewer than eight verified fresh disjoint trajectories are available')
    training.extend(fresh[:2])
    validation.extend(fresh[2:])
    train_ids = {source.original['layout_id'] for source in training}
    val_ids = {source.original['layout_id'] for source in validation}
    protected = set(snapshot['protected_layout_ids'])
    if len(train_ids) != 10 or len(val_ids) != 10 or train_ids & val_ids or (train_ids | val_ids) & protected:
        raise ValueError('Training/validation layouts overlap one another or a protected test cohort')
    training_manifest = TrainingManifest(
        version=3, description='10 offline training layouts; 3 distinct non-source recorded targets each; d/u unmeasured.',
        original_training_path=str(TRAINING.relative_to(ROOT)), original_training_sha256=sha256(TRAINING),
        exclusion_snapshot_sha256=sha256(snapshot_path), contexts=[training_context(source) for source in training],
        target_selection_rule=TARGET_SELECTION_RULE,
    )
    validation_manifest = ValidationManifest(
        version=3, split='component_validation_trajectory_positions',
        description='30 distinct recorded source states on 10 distinct validation layouts; 3 positions per layout.',
        count=30, distinct_layouts=10, positions=[0, 1, 2], exclusion_snapshot_sha256=sha256(snapshot_path),
        instances=validation_sources(validation),
    )
    destination = output / 'validation_sources_30.json'
    if destination.exists() and json.loads(destination.read_text()).get('distinct_layouts') == 4:
        historical = output / 'validation_sources_30_four_layouts.json'
        if historical.exists() and historical.read_bytes() != destination.read_bytes():
            raise ValueError('Historical four-layout manifest already exists with different contents')
        if not historical.exists():
            historical.write_bytes(destination.read_bytes())
        destination.unlink()
    old_training = output / 'training_candidates.json'
    if old_training.exists() and json.loads(old_training.read_text()).get('version') == 1:
        # Authorized preflight correction before any model/calibration manifest was prepared.
        archive = output / 'preflight_fixed_offsets'
        archive.mkdir(exist_ok=True)
        for name in ('training_candidates.json', 'validation_sources_30.json', 'dataset_provenance_checks.json'):
            old = output / name
            write_once(archive / name, json.loads(old.read_text()))
        for name in ('training_candidates.json', 'validation_sources_30.json', 'dataset_provenance_checks.json'):
            (output / name).unlink()
    write_once(output / 'training_candidates.json', training_manifest)
    write_once(destination, validation_manifest)
    write_once(output / 'dataset_provenance_checks.json', {
        'version': 3, 'training_layouts': 10, 'validation_layouts': 10, 'validation_source_states': 30,
        'training_candidates': 30, 'preferred_training_offsets': [1, 2, 4], 'validation_positions': [0, 1, 2],
        'actual_training_offsets': sorted({candidate['k'] for context in training_manifest['contexts']
                                           for candidate in context['candidates']}),
        'target_selection_rule': TARGET_SELECTION_RULE,
        'source_equal_training_targets': 0, 'duplicate_training_targets_within_context': 0,
        'reused_training_layouts': 8, 'reused_validation_layouts': 4, 'new_training_layouts': 2,
        'new_validation_layouts': 6, 'train_validation_intersections': 0, 'protected_intersections': 0,
        'fresh_history_intersections': 0, 'all_selected_trajectories_replayed': True,
        'source_path': PART1, 'source_sha256': snapshot['source_sha256'],
        'fresh_source_indices_in_selection_order': [source.original['source_index'] for source in fresh],
        'rejected_source_indices': rejected, 'exclusion_snapshot_sha256': sha256(snapshot_path),
        'training_manifest_sha256': sha256(output / 'training_candidates.json'),
        'validation_manifest_sha256': sha256(destination),
        'policy_relative_labels': 'Not supplied; require new execution calibration with frozen original policy.',
        'validation_contract': 'Source positions and provenance only; no future targets or action witnesses.',
    })
    print(f'Prepared/verified 10 training layouts, 30 targets; 10 validation layouts, 30 sources: {output}')


def main() -> None:
    """Prepare deterministic, inspectable offline inputs without invoking a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output.expanduser().resolve())


if __name__ == '__main__':
    main()
