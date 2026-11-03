"""Export compact portable launch inputs without reading held-out action witnesses."""
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import cast

from carl.icl.boards import canonical_layout_id, from_rows, level_key, state_hash
from carl.icl.checkpoints.io import load_generator_checkpoint
from carl.icl.config import ROOT, ExperimentConfig, load_config
from carl.icl.datasets import PRIVATE_FIELDS, PUBLIC_INSTANCE_FIELDS
from carl.icl.types import JSONObject
from scripts.generator_probe.prepare_sources import sha256
from scripts.goal_probe.data import OUTPUT, SOURCE

ASSETS = ROOT / 'configs/icl/checkpoints/launch100/assets'
DATA = ROOT / 'data/icl/campaign_dataset_12x12_v1'
VERBAL_ID = '3907728c160d892bf5763a7054b9729c05244562491854905d5b984ff6a2091a'
GROUNDED_ID = '7bfd933812e61e587cf83225d94519513c2de4cc1ce16c9f761e837f7bc709d4'


def relative(path: Path) -> str:
    """Repository-relative identity, without resolving an archival external symlink."""
    return str(path.relative_to(ROOT))


def read(path: Path) -> JSONObject:
    """Read selected public or training JSON; never emit boards."""
    return cast(JSONObject, json.loads(path.read_text()))


def serialized(document: object) -> bytes:
    """Canonical inspectable encoding for newly derived portable records."""
    return (json.dumps(document, indent=2, sort_keys=True) + '\n').encode()


def write(path: Path, content: bytes) -> None:
    """Never replace different immutable launch evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Existing portable artifact differs: {path}')
    else:
        path.write_bytes(content)


def reject_private(value: object) -> None:
    """Public cohort metadata must contain no actions, future endpoints or witnesses."""
    if isinstance(value, dict):
        if set(value) & PRIVATE_FIELDS:
            raise ValueError('Private witness field in public evaluation manifest')
        for item in value.values():
            reject_private(item)
    elif isinstance(value, list):
        for item in value:
            reject_private(item)


def cohort_checks() -> JSONObject:
    """Verify all 100 public identities and source metadata without reading solutions."""
    path = DATA / 'real_12x12x4/pilot100.json'
    public = read(path)
    reject_private(public)
    campaign = read(DATA / 'campaign_dataset_manifest.json')
    if sha256(path) != campaign['manifest_sha256']['real_12x12x4/pilot100.json']:
        raise ValueError('Existing frozen pilot identity differs')
    records = public['instances']
    ids: set[str] = set()
    for row in records:
        if set(row) - PUBLIC_INSTANCE_FIELDS or row['family'] != 'real_12x12x4' or row['split'] != 'pilot100':
            raise ValueError('Unexpected public pilot metadata')
        state = from_rows(row['state'])
        if (state.shape != (12, 12, 7) or state_hash(state) != row['state_hash']
                or canonical_layout_id(state) != row['layout_id'] or level_key(state) != row['level_key']):
            raise ValueError('Public board dimensions or identity checksum mismatch')
        ids.add(row['layout_id'])
    if len(records) != 100 or len(ids) != 100:
        raise ValueError('Launch requires exactly 100 distinct real layouts')
    train = read(SOURCE / 'training_candidates.json')['contexts']
    train_ids = {row['layout_id'] for row in train}
    validation_ids = {row['layout_id'] for row in read(SOURCE / 'validation_sources_30.json')['instances']}
    for original in (DATA / 'real_12x12x4/component_validation.json', DATA / 'real_12x12x4/scale_validation.json',
                     ROOT / 'data/icl/campaign_policy_validation_v1/real_12x12x4/component_validation.json'):
        validation_ids.update(row['layout_id'] for row in read(original)['instances'])
    if ids & train_ids or ids & validation_ids or train_ids & validation_ids:
        raise ValueError('Pilot, demonstration or validation layouts overlap')
    index = ROOT / 'data/icl/.private_campaign_shards.sqlite'
    with sqlite3.connect(f'file:{index}?mode=ro', uri=True) as database:
        for row in records:
            indexed = database.execute(
                'SELECT shards.sha256, boards.layout_id, boards.state_hash FROM boards JOIN shards '
                'ON boards.path=shards.path WHERE boards.path=? AND trajectory_index=?',
                (row['source_path'], row['source_index'])).fetchone()
            if indexed != (row['source_sha256'], row['layout_id'], row['state_hash']):
                raise ValueError('Pilot provenance does not match the existing source metadata index')
    return {'pilot_count': 100, 'unique_layouts': 100, 'family': 'real_12x12x4',
            'training_layouts_checked': len(train_ids), 'validation_layouts_checked': len(validation_ids),
            'pilot_train_intersections': 0, 'pilot_validation_intersections': 0,
            'public_witness_fields': 0, 'source_metadata_index_rows_verified': 100,
            'source_shards': len({row['source_path'] for row in records}),
            'source_metadata_index_sha256': sha256(index),
            'source_verification_scope': 'Existing indexed shard digest, trajectory index, initial state hash and layout ID. '
                                         'No trajectories or witness files opened; no new source-shard replay.',
            'pilot_manifest_sha256': sha256(path), 'training_layout_ids': sorted(train_ids),
            'validation_layout_ids': sorted(validation_ids), 'pilot_layout_ids': sorted(ids)}


def verbal_bundle(checkpoint_path: str) -> JSONObject:
    """Keep measured semantic examples and frozen controller; replace only generator prompt."""
    original_path = OUTPUT / 'verbal' / VERBAL_ID / 'checkpoint.json'
    calibration_path = original_path.parent / 'calibration' / VERBAL_ID / 'calibration.json'
    grounded_path = OUTPUT / 'verbal_grounded_retry' / GROUNDED_ID / 'checkpoint.json'
    original, calibration, grounded = read(original_path), read(calibration_path), read(grounded_path)
    if calibration['status'] != 'completed' or len(calibration['examples']) != 6:
        raise ValueError('Expected six completed measured semantic calibration examples')
    reference = original['training']['provenance']['reference']
    original_reference_config = reference['config'].copy()
    reference['config']['generator_checkpoint'] = checkpoint_path
    return {'schema': 'carl-verbal-launch-v1', 'calibration_status': calibration['status'],
            'config': original['config'], 'prompts': {**original['prompts'], 'generator': grounded['spec']['generator_prompt']},
            'examples': calibration['examples'], 'demonstrations': original['demonstrations'],
            'generator_examples': original['generator_examples'], 'reference': reference,
            'provenance': {'source_signature': VERBAL_ID, 'grounded_generator_signature': GROUNDED_ID,
                           'reference_policy_fingerprint': original['reference_policy_fingerprint'],
                           'calibration_contexts_with_hits': calibration['contexts_with_hits'],
                           'calibration_trials': calibration['trials'],
                           'original_reference_config': original_reference_config,
                           'original_reference_config_sha256': hashlib.sha256(serialized(original_reference_config)).hexdigest(),
                           'relocated_generator_checkpoint': checkpoint_path,
                           'source_files': {relative(p): sha256(p) for p in (original_path, calibration_path, grounded_path)},
                           'note': 'Training only; original controller/judge and measured labels retained. '
                                   'Generator prompt changed after observed validation failures; not independent retest.'}}


def export() -> None:
    """Freeze runtime dependencies and a content/provenance manifest, without an API."""
    checks = cohort_checks()
    config = load_config('configs/icl/prompts/generator_goal_progress.json')
    config = ExperimentConfig.model_validate({**config.model_dump(),
        'data_dir': relative(ASSETS / 'data'), 'generator_checkpoint': relative(ASSETS / 'checkpoint/generator.json'),
        'family': 'real_12x12x4', 'split': 'pilot100', 'instances': 100, 'easy': False,
        'scales': [0.85, 0.5], 'action_budget': 4000, 'max_expansions': 500,
        'repeat_seeds': [11], 'output_dir': 'analysis/luna_icl/launch100'})
    checkpoint_source = SOURCE / 'checkpoint_balanced/generator.json'
    checkpoint = load_generator_checkpoint(checkpoint_source, provider='openai')
    if len(checkpoint.examples) != 26:
        raise ValueError('Expected frozen measured 26-example generator')
    copies = {'data/training.json': DATA / 'training.json',
              'data/value_examples.json': DATA / 'value_examples.json',
              'data/campaign_dataset_manifest.json': DATA / 'campaign_dataset_manifest.json',
              'data/real_12x12x4/pilot100.json': DATA / 'real_12x12x4/pilot100.json',
              'checkpoint/generator.json': checkpoint_source,
              f'checkpoint/{checkpoint.calibration_evidence.path}': checkpoint_source.parent / checkpoint.calibration_evidence.path,
              'demonstrations.json': OUTPUT / 'data_training_demonstrations.json'}
    files: JSONObject = {}
    for destination, source in copies.items():
        content = source.read_bytes()
        write(ASSETS / destination, content)
        files[destination] = {'sha256': sha256(ASSETS / destination), 'bytes': len(content),
                              'source_path': relative(source), 'source_sha256': sha256(source), 'transform': 'byte-identical'}
    derived = {'config.json': config.model_dump(mode='json'),
               'verbal.json': verbal_bundle(relative(ASSETS / 'checkpoint/generator.json')),
               'cohort_checks.json': checks}
    for destination, document in derived.items():
        content = serialized(document)
        write(ASSETS / destination, content)
        files[destination] = {'sha256': sha256(ASSETS / destination), 'bytes': len(content),
                              'source_path': relative(Path(__file__).resolve()),
                              'source_sha256': sha256(Path(__file__).resolve()), 'transform': 'derived; see artifact provenance'}
    manifest = {'schema': 'carl-launch100-assets-v1', 'files': files, 'checks': checks,
                'policy_fingerprint': checkpoint.policy_fingerprint,
                'generator_examples': 26, 'unlabelled_task_progress_demonstrations': 6,
                'verbal_measured_examples': 6, 'new_api_calls': 0,
                'portability': 'Runtime uses included files only. Original absolute training signature source is archival '
                               'metadata retained exactly because it contributes to the frozen policy fingerprint; '
                               'do not normalize it or require its local source path to exist.',
                'source_config': 'configs/icl/prompts/generator_goal_progress.json',
                'source_config_resolved_sha256': hashlib.sha256(serialized(load_config(
                    'configs/icl/prompts/generator_goal_progress.json').model_dump(mode='json'))).hexdigest()}
    write(ASSETS / 'manifest.json', serialized(manifest))
    print(json.dumps({'files': len(files), 'bytes': sum(row['bytes'] for row in files.values()),
                      'pilot_count': checks['pilot_count'], 'unique_layouts': checks['unique_layouts']}))


if __name__ == '__main__':
    export()
