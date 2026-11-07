"""Deterministic paired context blocks and frozen cohort prefix invariants."""
import random
from collections import Counter
from pathlib import Path

import pytest

from scripts.figure1_pilot.block_order import ORDER_VERSION, order_jobs
from scripts.reachability_runner.engine import data_inputs, jobs, load_study


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*_args, **_kwargs):
        pytest.fail('Network access during offline ordering tests')

    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def synthetic(contexts, offsets=(1, 2), reps=2):
    result = []
    for ctx in contexts:
        for k in offsets:
            for arm in ('luna', 'terra_6_1'):
                for repetition in range(reps):
                    key = f'{ctx}-{k}-{arm}-{repetition}'
                    result.append({'key': key, 'seed': len(result) + 1, 'group': 'main', 'context': ctx,
                        'layout': f'layout-{ctx}', 'family': 'real_12x12x4', 'source_hash': f'source-{ctx}',
                        'target_hash': f'target-{ctx}-{k}', 'pair': f'{ctx}-{k}', 'k': k,
                        'arm': arm, 'repetition': repetition, 'model': f'model-{arm}'})
    return result


def schedule(rows, contexts, offsets=(1, 2), reps=2):
    return order_jobs(rows, main_contexts=contexts, pilot_contexts=(), reference_contexts=(),
                      core_offsets=offsets, arms=('luna', 'terra_6_1'), core_repetitions=reps,
                      reference_repetitions=1)


def test_blocks_are_paired_complete_and_independent_of_input_order():
    rows = synthetic(['b', 'a', 'c'])
    actual = schedule(rows, ['c', 'a', 'b'])
    shuffled = list(rows)
    random.Random(42).shuffle(shuffled)
    assert schedule(shuffled, ['c', 'a', 'b']) == actual
    assert ORDER_VERSION == 'paired-context-blocks-v1'
    assert [row['context'] for row in actual] == ['c'] * 8 + ['a'] * 8 + ['b'] * 8
    first = actual[:8]
    assert [(row['repetition'], row['k'], row['arm']) for row in first] == [
        (0, 1, 'luna'), (0, 1, 'terra_6_1'), (0, 2, 'terra_6_1'), (0, 2, 'luna'),
        (1, 1, 'terra_6_1'), (1, 1, 'luna'), (1, 2, 'luna'), (1, 2, 'terra_6_1')]
    assert {row['key'] for row in actual} == {row['key'] for row in rows}
    assert {row['seed'] for row in actual} == {row['seed'] for row in rows}


def test_incomplete_or_inconsistent_grid_is_rejected():
    rows = synthetic(['one'])
    with pytest.raises(ValueError, match='missing paired'):
        schedule(rows[:-1], ['one'])
    with pytest.raises(ValueError, match='identities'):
        schedule(rows + [rows[0]], ['one'])
    altered = [dict(row) for row in rows]
    altered[0]['target_hash'] = 'wrong'
    with pytest.raises(ValueError, match='exactly one k and target'):
        schedule(altered, ['one'])
    with pytest.raises(ValueError, match='selected context'):
        schedule([], ['one'])


def test_reference_blocks_have_both_arms_and_expected_target_count():
    raw = synthetic(['ref'])
    for row in raw:
        row['group'] = 'calibration_A'
    duplicate = [{**row, 'group': 'calibration_B', 'key': 'b-' + row['key'], 'seed': row['seed'] + 100} for row in raw]
    result = order_jobs(raw + duplicate, main_contexts=(), pilot_contexts=(), reference_contexts=('ref',),
                        core_offsets=(1, 2), arms=('terra_6_1', 'luna'), core_repetitions=1,
                        reference_repetitions=2, reference_targets_by_context={'ref': 2})
    assert Counter(row['group'] for row in result) == {'calibration_A': 8, 'calibration_B': 8}
    with pytest.raises(ValueError, match='preregistered targets'):
        order_jobs(raw + duplicate, main_contexts=(), pilot_contexts=(), reference_contexts=('ref',),
                   core_offsets=(1, 2), arms=('luna', 'terra_6_1'), core_repetitions=1,
                   reference_repetitions=2, reference_targets_by_context={'ref': 3})


@pytest.mark.skipif(
    not Path("data/icl/reachability_exploratory_v2/private/source_bundle.tar.gz").is_file(),
    reason="requires the local trajectory witness bundle",
)
def test_real_n100_context_order_has_exact_n50_prefix():
    study = load_study(Path('configs/icl/reachability_runner/offline_core_plan.json'))
    study.n_pilot = 0
    study.reference_contexts = 0
    study.reference_targets = 0
    dataset, _training, core, reference, _boards = data_inputs(study)
    study.n_main = 100
    hundred = jobs(study, dataset, core, reference, 'same-explicit-contract-for-order-test')
    study.n_main = 50
    fifty = jobs(study, dataset, core, reference, 'same-explicit-contract-for-order-test')
    def order(rows, n):
        return order_jobs(rows, main_contexts=dataset['n100'][:n], pilot_contexts=(), reference_contexts=(),
                          core_offsets=study.core_offsets, arms=tuple(study.arms), core_repetitions=5,
                          reference_repetitions=10)
    ordered100, ordered50 = order(hundred, 100), order(fifty, 50)
    assert len(ordered100) == 6000 and len(ordered50) == 3000
    assert ordered100[:3000] == ordered50
    assert len({row['layout'] for row in ordered100[:3000]}) == 50
    assert len({row['layout'] for row in ordered100[3000:]}) == 50


@pytest.mark.skipif(
    not Path("data/icl/reachability_exploratory_v2/private/source_bundle.tar.gz").is_file(),
    reason="requires the local trajectory witness bundle",
)
def test_real_reference_contexts_are_not_intermingled_with_core():
    study = load_study(Path('configs/icl/reachability_runner/offline_core_plan.json'))
    dataset, _training, core, reference, _boards = data_inputs(study)
    rows = jobs(study, dataset, core, reference, 'offline-order-reference-check')
    contexts = dataset['calibration_contexts'][:study.reference_contexts]
    target_counts = Counter(pair['context_id'] for pair in reference)
    result = order_jobs(rows, main_contexts=dataset['n100'][:study.n_main], pilot_contexts=(),
                        reference_contexts=contexts, core_offsets=study.core_offsets, arms=tuple(study.arms),
                        core_repetitions=study.core_repetitions,
                        reference_repetitions=study.reference_repetitions_per_group,
                        reference_targets_by_context={ctx: min(study.reference_targets, target_counts[ctx]) for ctx in contexts})
    assert len(result) == 5560
    assert Counter(row['group'] for row in result) == {'main': 3000, 'calibration_A': 1280, 'calibration_B': 1280}
    assert all(row['group'] == 'main' for row in result[:3000])
    assert all(row['group'] == 'calibration_A' for row in result[3000:4280])
