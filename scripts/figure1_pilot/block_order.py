"""Deterministic, paired scheduling for a frozen Figure 1 trial manifest.

This only reorders existing trials: identities, streams, pairs and action limits
are never changed. A new order is a contract change and must be fingerprinted.
"""
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

ORDER_VERSION = 'paired-context-blocks-v1'
CORE_STAGES = ('pilot', 'main')
REFERENCE_STAGES = ('calibration_A', 'calibration_B')


def _unique_sequence(values: Sequence[object], what: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate {what} in ordering contract')


def _block(block: list[dict[str, Any]], *, arms: tuple[str, ...],
           offsets: tuple[int, ...] | None, repetitions: int, block_index: int) -> list[dict[str, Any]]:
    """Check and interleave all paired model/k/repetition combinations."""
    if not block:
        return []
    if len({r['layout'] for r in block}) != 1 or len({r['family'] for r in block}) != 1:
        raise ValueError('A context block must have one family and canonical layout')
    if len({r['source_hash'] for r in block}) != 1:
        raise ValueError('A context block must have one common source')
    # A calibration pair is one target. Multiple core k values can alias the
    # same target but remain distinct pair IDs and must not be deduplicated.
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in block:
        by_pair[row['pair']].append(row)
    by_offset: dict[int, str] = {}
    for pair, rows in by_pair.items():
        k = rows[0]['k']
        if any(row['k'] != k or row['target_hash'] != rows[0]['target_hash'] for row in rows):
            raise ValueError('A pair must identify exactly one k and target')
        if k in by_offset:
            raise ValueError('Distinct pairs in a block cannot share an offset')
        by_offset[k] = pair
    selected_offsets = offsets if offsets is not None else tuple(sorted(by_offset))
    if set(selected_offsets) != set(by_offset):
        raise ValueError('A core context does not contain the complete frozen k grid')
    indexed: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in block:
        if row['arm'] not in arms:
            raise ValueError('Unexpected model arm in trial block')
        key = (row['k'], row['arm'], row['repetition'])
        if key in indexed:
            raise ValueError('Duplicate paired trial in block')
        indexed[key] = row
    expected = {(k, arm, repetition) for k in selected_offsets for arm in arms for repetition in range(repetitions)}
    if set(indexed) != expected:
        raise ValueError('A context block is missing paired arms, offsets or repetitions')
    result = []
    for repetition in range(repetitions):
        for offset_index, k in enumerate(selected_offsets):
            rotation = (block_index + repetition + offset_index) % len(arms)
            ordered_arms = arms[rotation:] + arms[:rotation]
            result.extend(indexed[(k, arm, repetition)] for arm in ordered_arms)
    return result


def order_jobs(jobs: list[dict[str, Any]], *, main_contexts: Sequence[str],
               pilot_contexts: Sequence[str], reference_contexts: Sequence[str],
               core_offsets: Sequence[int], arms: Sequence[str], core_repetitions: int,
               reference_repetitions: int,
               reference_targets_by_context: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Schedule context blocks in preregistered order, pairing both arms and all k.

    Blocks run pilot, then main in the supplied frozen N100 order, then reference
    group A and B in the supplied context order. Each core context contains the
    same k x repetitions x arms before the next context. Arm order rotates by
    context, k and repetition. An unfinished trial retains its position on resume.
    A N100 study always starts with the original first 50 contexts; separate N50
    and N100 contracts still have different trial identities and random streams.
    """
    main_contexts, pilot_contexts, reference_contexts = map(tuple, (main_contexts, pilot_contexts, reference_contexts))
    arms, core_offsets = tuple(sorted(arms)), tuple(core_offsets)
    for values, name in ((main_contexts, 'main contexts'), (pilot_contexts, 'pilot contexts'),
                         (reference_contexts, 'reference contexts'), (arms, 'arms'), (core_offsets, 'offsets')):
        _unique_sequence(values, name)
    if len(arms) != 2 or not core_offsets or core_repetitions < 1 or reference_repetitions < 1:
        raise ValueError('Expected two arms, offsets and positive repetition counts')
    if len({r['key'] for r in jobs}) != len(jobs) or len({r['seed'] for r in jobs}) != len(jobs):
        raise ValueError('Trial identities and action seeds must be unique')
    contexts = {'pilot': pilot_contexts, 'main': main_contexts,
                'calibration_A': reference_contexts, 'calibration_B': reference_contexts}
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in jobs:
        stage, context = row['group'], row['context']
        if stage not in contexts or context not in contexts[stage]:
            raise ValueError('Trial refers to an unselected stage or context')
        by_group[(stage, context)].append(row)
    result: list[dict[str, Any]] = []
    for stage in CORE_STAGES + REFERENCE_STAGES:
        for index, context in enumerate(contexts[stage]):
            block = by_group.get((stage, context), [])
            # Selected contexts must never silently disappear from the grid.
            if not block:
                raise ValueError('A selected context has no trials')
            if stage in REFERENCE_STAGES and reference_targets_by_context is not None:
                expected_targets = reference_targets_by_context.get(context)
                if expected_targets is None or len({row['pair'] for row in block}) != expected_targets:
                    raise ValueError('Reference context is missing preregistered targets')
            result.extend(_block(block, arms=arms, offsets=core_offsets if stage in CORE_STAGES else None,
                                 repetitions=core_repetitions if stage in CORE_STAGES else reference_repetitions,
                                 block_index=index))
    if len(result) != len(jobs):
        raise ValueError('Ordering discarded trials')
    return result
