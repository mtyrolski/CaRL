"""Prepare 30 offline validation states on four already exposed layouts.

Only the original component-validation trajectories and their private validation
witnesses are inspected. No model calls or new layouts are involved. The public
manifest contains source states and provenance, never actions or future targets.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TypedDict, cast

import joblib
import numpy as np

from carl.icl.boards import canonical_layout_id, from_rows, state_hash, to_rows
from carl.icl.config import ROOT
from carl.icl.datasets import verify_solution
from carl.icl.types import Board, BoardRows

SOURCE_MANIFEST = ROOT / 'data/icl/campaign_dataset_v2/real_12x12x4/component_validation.json'
WITNESSES = ROOT / 'data/icl/campaign_dataset_v2_private/component_validation/real_12x12x4.json'
OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/generator_probe_v2/validation_sources_30_four_layouts.json'


class OriginalSource(TypedDict):
    """Fields consumed from an original public validation record."""

    id: str
    board_id: str
    trajectory_id: str
    source_path: str
    source_index: int
    source_sha256: str
    state: BoardRows
    state_hash: str
    layout_id: str


class SourceState(OriginalSource):
    """A source at a recorded trajectory position, without its continuation."""

    position: int


class ValidationWitness(TypedDict):
    """Private validation-only actions; never included in the public output."""

    board_id: str
    actions: list[int]


class SourceManifest(TypedDict):
    """Reproducible source states and their complete public lineage."""

    version: int
    split: str
    description: str
    selection: str
    source_manifest_path: str
    source_manifest_sha256: str
    private_validation_witness_sha256: str
    count: int
    distinct_layouts: int
    positions_per_original_board: dict[str, int]
    instances: list[SourceState]


def sha256(path: Path) -> str:
    """Hash an input file without emitting its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_sources() -> list[tuple[OriginalSource, list[Board]]]:
    """Match only approved validation trajectories to their original witnesses."""
    originals = cast(list[OriginalSource], json.loads(SOURCE_MANIFEST.read_text())['instances'])
    witnesses = cast(list[ValidationWitness], json.loads(WITNESSES.read_text())['witnesses'])
    if len(originals) != 4 or len({row['layout_id'] for row in originals}) != 4:
        raise ValueError('Expected exactly four distinct original validation layouts')
    by_id = {row['board_id']: row['actions'] for row in witnesses}
    if len(by_id) != len(witnesses) or set(by_id) != {row['board_id'] for row in originals}:
        raise ValueError('Original validation witness identities differ from the manifest')
    shards: dict[str, list[Board]] = {}
    digests: dict[str, str] = {}
    verified = []
    for original in originals:
        path = ROOT / original['source_path']
        if original['source_path'] not in shards:
            digests[original['source_path']] = sha256(path)
            raw = joblib.load(path)
            shards[original['source_path']] = list(raw.values()) if isinstance(raw, dict) else list(raw)
        if digests[original['source_path']] != original['source_sha256']:
            raise ValueError('Offline source checksum differs from the original manifest')
        trajectory = np.asarray(shards[original['source_path']][original['source_index']])
        if to_rows(trajectory[0]) != original['state'] or state_hash(trajectory[0]) != original['state_hash']:
            raise ValueError('Offline source initial state differs from the original manifest')
        states = verify_solution(from_rows(original['state']), by_id[original['board_id']])
        if len(states) != len(trajectory) or any(
            not np.array_equal(state, recorded)
            for state, recorded in zip(states, trajectory, strict=True)
        ):
            raise ValueError('Original validation witness does not reproduce the recorded trajectory')
        if len(states) < 8 or any(canonical_layout_id(state) != original['layout_id'] for state in states):
            raise ValueError('Validation trajectory is too short or changes its fixed layout')
        verified.append((original, states))
    return verified


def prepare_manifest() -> SourceManifest:
    """Select round-robin positions 0..7, stopping after exactly 30 sources."""
    originals = verified_sources()
    instances: list[SourceState] = []
    for position in range(8):
        for original, trajectory in originals:
            if len(instances) == 30:
                break
            state = trajectory[position]
            instances.append(SourceState(
                id=f"{original['id']}:position_{position:02d}",
                board_id=original['board_id'], trajectory_id=original['trajectory_id'],
                source_path=original['source_path'], source_index=original['source_index'],
                source_sha256=original['source_sha256'], position=position,
                state=to_rows(state), state_hash=state_hash(state), layout_id=original['layout_id'],
            ))
    if len(instances) != 30 or len({row['state_hash'] for row in instances}) != 30:
        raise ValueError('Expected exactly 30 distinct source states')
    counts = dict(Counter(row['board_id'] for row in instances))
    if list(counts.values()) != [8, 8, 7, 7]:
        raise ValueError('Round-robin source counts differ from the declared design')
    return SourceManifest(
        version=1, split='component_validation_trajectory_positions',
        description='30 distinct recorded states on 4 existing validation layouts; not 30 independent boards.',
        selection='Original manifest order; positions 0..7 round-robin; stop at 30; no replacement or target selection.',
        source_manifest_path=str(SOURCE_MANIFEST.relative_to(ROOT)),
        source_manifest_sha256=sha256(SOURCE_MANIFEST),
        private_validation_witness_sha256=sha256(WITNESSES),
        count=30, distinct_layouts=4, positions_per_original_board=counts, instances=instances,
    )


def main() -> None:
    """Write once, or verify that an existing source manifest is identical."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = prepare_manifest()
    output = args.output.expanduser().resolve()
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError('Existing manifest differs; refusing to overwrite it')
        print(f'Verified existing manifest: {output}')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as file:
            file.write(json.dumps(manifest, indent=2) + '\n')
        print(f'Prepared 30 source states on 4 layouts: {output}')


if __name__ == '__main__':
    main()
