"""Checks for a separately assigned, D4-disjoint campaign training cohort."""

import argparse
import json
import shutil
import sqlite3

import joblib
import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from scripts import prepare_overnight_cpu_stage02_train as training
from scripts.prepare_overnight_cpu_stage02 import atomic_json
from scripts.prepare_overnight_cpu_stage02 import file_sha


def state(player_column: int) -> np.ndarray:
    cells = np.ones((12, 12), dtype=np.int8)
    cells[0, :] = cells[-1, :] = 0
    cells[:, 0] = cells[:, -1] = 0
    cells[1, 1:5] = 2
    cells[3, 1:5] = 4
    cells[5, player_column] = 5
    return np.eye(7, dtype=np.float32)[cells]


def test_reservation_excludes_fresh_and_frozen_history_and_stage01():
    history = {'observed_layout_ids': ['fresh_seen'], 'previously_protected_layout_ids': ['fresh_protected']}
    frozen = {'observed_layout_ids': ['old_seen'], 'training_prohibited_layout_ids': ['frozen_protected']}
    trajectories = [{'layout_id': 'development'}, {'layout_id': 'evaluation'}]
    assert training.excluded_layouts(history, frozen, trajectories) == {
        'fresh_seen', 'fresh_protected', 'old_seen', 'frozen_protected', 'development', 'evaluation',
    }


def test_cohort_rejects_overlap_duplicate_or_changed_index():
    training.verify_cohort(((3, 'layout_a'), (4, 'layout_b')), {'protected'}, {3: 'layout_a', 4: 'layout_b'})
    with pytest.raises(ValueError, match='intersects'):
        training.verify_cohort(((3, 'layout_a'),), {'layout_a'}, {3: 'layout_a'})
    with pytest.raises(ValueError, match='duplicate'):
        training.verify_cohort(((3, 'layout_a'), (4, 'layout_a')), set(), {3: 'layout_a', 4: 'layout_a'})
    with pytest.raises(ValueError, match='index'):
        training.verify_cohort(((3, 'layout_a'),), set(), {3: 'different'})


def test_training_role_overrides_development_evaluation_defaults():
    states = [state(1), state(2)]
    board, pairs, missing = training.training_records(7, 0, states, [3], layout=canonical_layout_id(states[0]),
                                                       source_hash='sha', seed=20261115)
    assert board['split'] == 'train'
    assert board['purpose'] == 'training_only'
    assert board['training_eligibility'] == 'campaign_train_assigned'
    assert pairs[0]['split'] == 'train' and pairs[0]['k'] == 1
    assert pairs[0]['demonstration_actions'] == [3]
    assert {item['k'] for item in missing} == {2, 4, 8, 16, 32}


def test_history_hash_matches_atomic_serialisation(tmp_path):
    history = {'observed_layout_ids': ['a'], 'previously_protected_layout_ids': ['b']}
    atomic_json(tmp_path / 'history.json', history)
    assert file_sha(tmp_path / 'history.json') == training.json_sha(history)


def test_small_preparation_saves_bound_exposure_and_reuses_shards(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    source, alias = root / training.SOURCE_RELATIVE, root / training.ALIAS_RELATIVE
    source.parent.mkdir(parents=True)
    alias.parent.mkdir(parents=True)
    states = [state(1), state(2)]
    joblib.dump([states], source)
    shutil.copyfile(source, alias)
    layout = canonical_layout_id(states[0])
    index = root / training.INDEX_RELATIVE
    index.parent.mkdir(parents=True)
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE shards (path TEXT, sha256 TEXT)')
        db.execute('CREATE TABLE boards (path TEXT, trajectory_index INTEGER, layout_id TEXT)')
        db.execute('INSERT INTO shards VALUES (?, ?)', (training.SOURCE_RELATIVE, file_sha(source)))
        db.execute('INSERT INTO boards VALUES (?, ?, ?)', (training.SOURCE_RELATIVE, 0, layout))
    stage01 = tmp_path / 'stage01'
    stage01.mkdir()
    frozen = {'observed_layout_ids': [], 'training_prohibited_layout_ids': ['reserved'],
              'development_layout_ids': ['development'], 'campaign_evaluation_layout_ids': ['reserved']}
    atomic_json(stage01 / 'exposure_register.json', frozen)
    (stage01 / 'trajectories.jsonl').write_text(json.dumps({'layout_id': 'development'}) + '\n')
    atomic_json(stage01 / 'data_manifest.json', {'outputs_sha256': {
        name: file_sha(stage01 / name) for name in ('exposure_register.json', 'trajectories.jsonl')}})
    history = {'canonicalization': 'carl-layout-d4-v1', 'observed_layout_ids': [],
               'previously_protected_layout_ids': [], 'sources': [], 'limitations': []}
    monkeypatch.setattr(training, 'ROOT', root)
    monkeypatch.setattr(training, 'COHORT', ((0, layout),))
    monkeypatch.setattr(training, 'history_register', lambda: history)
    monkeypatch.setattr(training, 'verified_trajectory', lambda _: (states, [3]))
    stage02_script = root / 'scripts/prepare_overnight_cpu_stage02.py'
    stage02_script.parent.mkdir()
    stage02_script.write_text('# fake hashed dependency for this unit test\n')
    args = argparse.Namespace(stage01_data=stage01, output=tmp_path / 'out', seed=20261115, workers=1, resume=False)
    first = training.prepare(args)
    assert first['state'] == 'complete' and first['replayed_boards'] == 1
    exposure = json.loads((args.output / 'exposure_register.json').read_text())
    assert exposure['train_layout_ids'] == [layout]
    assert exposure['old_checkpoint_exposure'] == 'unknown'
    assert exposure['stage01_exposure_sha256'] == file_sha(stage01 / 'exposure_register.json')
    assert exposure['fresh_history_sha256'] == file_sha(args.output / 'history_register.json')
    before = (args.output / 'candidates.jsonl').read_bytes()
    args.resume = True
    second = training.prepare(args)
    assert second['state'] == 'complete'
    assert (args.output / 'candidates.jsonl').read_bytes() == before
