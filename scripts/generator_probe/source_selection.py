"""Freeze exclusion identities and select new offline trajectories without APIs."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import TypedDict, cast

from carl.icl.config import ROOT
from scripts.datasets.audit_campaign_history import extract
from scripts.generator_probe.prepare_sources import sha256

DATA = ROOT / 'data/icl'
INDEX = DATA / '.private_campaign_shards.sqlite'
HISTORY = ROOT / 'analysis/luna_icl/campaign_100/history_snapshot.json'
PART1 = 'rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_1.pkl'
SEED = 2026110203
PUBLIC_NAMES = frozenset({'training.json', 'demonstrations.json', 'test.json', 'test_easy.json',
                         'validation.json', 'validation_easy.json', 'component_validation.json',
                         'scale_validation.json', 'pilot100.json', 'holdout1000.json'})


class ExclusionSnapshot(TypedDict):
    """Immutable selection-time input identity sets."""

    version: int
    selection_seed: int
    source_path: str
    source_sha256: str
    history_sha256: str
    public_manifest_sha256: dict[str, str]
    observed_request_sha256: dict[str, str]
    excluded_layout_ids: list[str]
    protected_layout_ids: list[str]


class IndexedSource(TypedDict):
    """Source identity and deterministic rank without a board or witness."""

    source_path: str
    source_sha256: str
    source_index: int
    layout_id: str
    state_hash: str
    rank: str


def index_rows() -> list[tuple[int, str, str, int, int]]:
    """Read only initial-state identities for the indexed original shard."""
    with sqlite3.connect(INDEX.resolve().as_uri() + '?mode=ro', uri=True) as db:
        row = db.execute('SELECT sha256 FROM shards WHERE path=?', (PART1,)).fetchone()
        if row is None or row[0] != sha256(ROOT / PART1):
            raise ValueError('Offline part-1 shard differs from its frozen cross-shard index')
        return db.execute('SELECT trajectory_index, layout_id, state_hash, trajectory_states, unsolved '
                          'FROM boards WHERE path=? ORDER BY trajectory_index', (PART1,)).fetchall()


def make_snapshot() -> ExclusionSnapshot:
    """Union history, current public splits, and saved request layout identities."""
    indices = {row[0]: row[1] for row in index_rows()}
    history = json.loads(HISTORY.read_text())
    excluded = set(history['observed_layout_ids']) | set(history['reserved_layout_ids'])
    protected: set[str] = set()
    manifests: dict[str, str] = {}
    for path in sorted(DATA.rglob('*.json')):
        if path.name not in PUBLIC_NAMES or any('private' in part or part == 'calibration' for part in path.parts):
            continue
        if path.name != 'training.json' and path.name != 'demonstrations.json' and path.parent.name != 'real_12x12x4':
            continue
        document = json.loads(path.read_text())
        rows = document.get('instances', document.get('contexts', []))
        layout_ids: set[str] = set()
        for row in rows:
            if 'layout_id' in row:
                layout_ids.add(row['layout_id'])
            else:
                match = re.fullmatch(r'trajectory_(\d+)(?:_one_move)?', row['id'])
                if match is None or int(match[1]) not in indices:
                    raise ValueError(f'Cannot resolve public record identity without inspecting a board: {path}')
                layout_ids.add(indices[int(match[1])])
        excluded.update(layout_ids)
        if path.name in {'test.json', 'test_easy.json', 'pilot100.json', 'holdout1000.json'}:
            protected.update(layout_ids)
        manifests[str(path.relative_to(ROOT))] = sha256(path)
    requests: dict[str, str] = {}
    for path in sorted((ROOT / 'analysis/luna_icl').glob('**/openai/*.json')):
        document = json.loads(path.read_text())
        if isinstance(document, dict) and 'request' in document:
            layouts: set[str] = set()
            old_indices: set[int] = set()
            extract(document['request'], layouts, old_indices)
            excluded.update(layouts)
            excluded.update(indices[index] for index in old_indices if index in indices)
            requests[str(path.relative_to(ROOT))] = sha256(path)
    return ExclusionSnapshot(
        version=1, selection_seed=SEED, source_path=PART1, source_sha256=sha256(ROOT / PART1),
        history_sha256=sha256(HISTORY), public_manifest_sha256=manifests,
        observed_request_sha256=requests, excluded_layout_ids=sorted(excluded),
        protected_layout_ids=sorted(protected),
    )


def exclusion_snapshot(path: Path) -> ExclusionSnapshot:
    """Freeze discovery once; subsequent model observations cannot change selection."""
    if path.exists():
        snapshot = cast(ExclusionSnapshot, json.loads(path.read_text()))
        if snapshot['selection_seed'] != SEED or snapshot['source_sha256'] != sha256(ROOT / PART1):
            raise ValueError('Saved selection parameters or offline source changed')
        if snapshot['history_sha256'] != sha256(HISTORY):
            raise ValueError('Original history snapshot changed')
        for filename, digest in snapshot['public_manifest_sha256'].items():
            if sha256(ROOT / filename) != digest:
                raise ValueError(f'Original public exclusion manifest changed: {filename}')
        return snapshot
    snapshot = make_snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as file:
        file.write(json.dumps(snapshot, indent=2) + '\n')
    return snapshot


def fresh_candidates(snapshot: ExclusionSnapshot) -> list[IndexedSource]:
    """Rank unexposed canonical layouts without loading their trajectories."""
    excluded = set(snapshot['excluded_layout_ids'])
    candidates: list[IndexedSource] = []
    for index, layout, state, length, unsolved in index_rows():
        if layout in excluded or length <= 16 or not unsolved:
            continue
        identity = f"{snapshot['source_sha256']}:{index}"
        rank = hashlib.sha256(f'generator-probe-v2:{SEED}:{identity}'.encode()).hexdigest()
        candidates.append(IndexedSource(source_path=PART1, source_sha256=snapshot['source_sha256'],
                          source_index=index, layout_id=layout, state_hash=state, rank=rank))
    candidates.sort(key=lambda row: (row['rank'], row['source_index']))
    unique = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate['layout_id'] not in seen:
            unique.append(candidate)
            seen.add(candidate['layout_id'])
    return unique
