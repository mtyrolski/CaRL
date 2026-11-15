#!/usr/bin/env python3
"""Freeze historical Qwen outputs without conflating their prompt protocols.

Original responses, including malformed ones, remain untouched. Only current
boards, goals and previously frozen support examples occur in prompts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import SYSTEM
from carl.icl.intent_generator_experiment import prompt_messages

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'analysis/intent_generator_20261114T032232Z'
TRANSFER = ROOT / 'analysis/lora_matrix_20261114T065804Z/eval_transfer'
SNAPSHOT = ROOT / 'analysis/lora_matrix_20261114T065804Z/code_snapshot'
REPRESENTATIONS = ROOT / 'analysis/intent_representations_20261114T075324Z'
MODEL = ORIGINAL / 'models/Qwen3-0.6B'
ADAPTER = ORIGINAL / 'adapter/checkpoint-220'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
MODEL_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
ADAPTER_SHA = 'e7cb661a4a0de1f43652b204ccd2b7bfa9a9a0a06f0e5c74a64ccb4c103b866e'
TOKENIZER_SHA = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'


class Cohort(StrEnum):
    LEGACY = 'legacy_transfer_40'
    REPRESENTATION = 'representation_128'


class Method(StrEnum):
    BASE = 'base'
    ICL = 'icl_legacy'
    ADAPTER = 'lora_validation_220'


class PredictionStatus(StrEnum):
    COMPLETE = 'complete'
    MISSING = 'missing'


COHORT_METHODS = {Cohort.LEGACY: (Method.BASE, Method.ICL, Method.ADAPTER),
                  Cohort.REPRESENTATION: (Method.BASE, Method.ICL)}
FAMILY_COUNTS = {'real_12x12x4': 4, 'proposal_gamma2_12x12x10': 10,
                 'generated_v1_12x12x10': 10}


def sha(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def by_record(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result = {row['record_id']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate record IDs in {name}')
    return result


def atomically(path: Path, text: str) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomically(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomically(path, ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
                             for row in rows))


def prompt_function_fingerprint(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prompt_messages']
    if len(functions) != 1:
        raise ValueError('Historical prompt source is missing or ambiguous')
    return hashlib.sha256(ast.dump(functions[0], include_attributes=False).encode()).hexdigest()


def read_selected_data(path: Path, ids_path: Path, expected_size: int) -> list[dict[str, Any]]:
    manifest = json_file(ids_path)
    if manifest.get('dataset_sha256') not in (None, sha(path)):
        raise ValueError(f'Frozen IDs refer to another dataset: {ids_path}')
    lookup = by_record(jsonl(path), str(path))
    ids = manifest['record_ids']
    if len(ids) != expected_size or len(set(ids)) != len(ids) or any(identifier not in lookup for identifier in ids):
        raise ValueError(f'Unexpected frozen context IDs in {ids_path}')
    rows = [lookup[identifier] for identifier in ids]
    if any(row['split'] != 'test' for row in rows) or len({row['layout_id'] for row in rows}) != expected_size:
        raise ValueError('Each cohort must use one frozen test context per layout')
    for row in rows:
        family = row.get('evaluation_family') or row.get('family')
        board = row['start_observation']
        if (family not in FAMILY_COUNTS or len(board) != 12 or any(len(line) != 12 for line in board)
                or sum(line.count('$') + line.count('*') for line in board) != FAMILY_COUNTS[family]):
            raise ValueError(f'Unexpected board family, size or box count: {row["record_id"]}')
    return rows


def source_files() -> dict[str, Path]:
    pred = REPRESENTATIONS / 'predictions'
    return {
        'transfer_data': TRANSFER / 'dataset.jsonl', 'transfer_ids': TRANSFER / 'test_ids.json',
        'transfer_manifest': TRANSFER / 'manifest.json', 'transfer_base': TRANSFER / 'base.jsonl',
        'transfer_base_run': TRANSFER / 'base.run.json', 'transfer_adapter': TRANSFER / 'r4_best220.jsonl',
        'transfer_adapter_run': TRANSFER / 'r4_best220.run.json',
        'legacy_icl': ORIGINAL / 'icl_followup/predictions.jsonl',
        'legacy_icl_run': ORIGINAL / 'icl_followup/predictions.run.json',
        'legacy_icl_prompts': ORIGINAL / 'icl_followup/predictions.prompts.jsonl',
        'legacy_support': ORIGINAL / 'icl_followup/support.json',
        'original_data': ORIGINAL / 'dataset.jsonl', 'adapter_best': ORIGINAL / 'adapter/best.json',
        'historical_prompt_source': SNAPSHOT / 'intent_generator_experiment.py',
        'historical_prompt_snapshot': SNAPSHOT / 'manifest.json',
        'representation_data': REPRESENTATIONS / 'data/dataset.jsonl',
        'representation_ids': REPRESENTATIONS / 'data/test_ids.json',
        'representation_manifest': REPRESENTATIONS / 'data/manifest.json',
        'representation_support': REPRESENTATIONS / 'support_manifest.json',
        'representation_base': pred / 'base_json_zero.jsonl',
        'representation_base_run': pred / 'base_json_zero.run.json',
        'representation_base_prompts': pred / 'base_json_zero.prompts.jsonl',
        'representation_icl': pred / 'base_json_legacy.jsonl',
        'representation_icl_run': pred / 'base_json_legacy.run.json',
        'representation_icl_prompts': pred / 'base_json_legacy.prompts.jsonl',
    }


def ensure_protocol(files: dict[str, Path]) -> dict[str, Any]:
    base_run = json_file(files['transfer_base_run'])
    adapted_run = json_file(files['transfer_adapter_run'])
    icl_run = json_file(files['legacy_icl_run'])
    system_sha = hashlib.sha256(SYSTEM.encode()).hexdigest()
    for run in (base_run, adapted_run, icl_run):
        if (run['revision'] != REVISION or run['decoding'] != 'greedy' or run['enable_thinking'] is not False
                or run['max_new_tokens'] != 128 or run['prompt_sha256'] != system_sha):
            raise ValueError('Legacy outputs use incompatible generator settings or system prompts')
    if (base_run['adapter'] is not None or adapted_run['adapter_weights_sha256'] != ADAPTER_SHA
            or icl_run['adapter'] is not None):
        raise ValueError('Legacy output adapter provenance differs')
    if (sha(MODEL / 'model.safetensors') != MODEL_SHA or sha(MODEL / 'tokenizer.json') != TOKENIZER_SHA
            or sha(ADAPTER / 'adapter_model.safetensors') != ADAPTER_SHA):
        raise ValueError('Cached model, tokenizer or selected checkpoint changed')
    if json_file(files['adapter_best'])['step'] != 220:
        raise ValueError('Checkpoint 220 is not the recorded validation selection')
    snapshot = json_file(files['historical_prompt_snapshot'])
    if sha(files['historical_prompt_source']) != snapshot['additional_inputs']['intent_generator_experiment.py']:
        raise ValueError('Historical prompt-source snapshot differs from its manifest')
    current = Path(inspect.getfile(prompt_messages))
    if prompt_function_fingerprint(current) != prompt_function_fingerprint(files['historical_prompt_source']):
        raise ValueError('Current and historical zero-shot prompt builders differ')
    if icl_run['prompts_sha256'] != sha(files['legacy_icl_prompts']):
        raise ValueError('Saved legacy ICL prompts differ from their metadata')
    if icl_run['support_sha256'] != sha(files['legacy_support']):
        raise ValueError('Legacy support manifest differs from its metadata')
    for method in ('base', 'icl'):
        run = json_file(files[f'representation_{method}_run'])
        if (run['revision'] != REVISION or run['model_weights_sha256'] != MODEL_SHA
                or run['tokenizer_sha256'] != TOKENIZER_SHA or run['representation'] != 'json'
                or run['decoding'] != 'greedy' or run['enable_thinking'] is not False
                or run['max_new_tokens'] != 128 or run['adapter'] is not None
                or run['data_sha256'] != sha(files['representation_data'])
                or run['test_ids_sha256'] != sha(files['representation_ids'])
                or run['support_manifest_sha256'] != sha(files['representation_support'])
                or run['prompts_sha256'] != sha(files[f'representation_{method}_prompts'])):
            raise ValueError(f'Representation {method} metadata mismatch')
    return {'model_revision': REVISION, 'model_weights_sha256': MODEL_SHA,
            'tokenizer_sha256': TOKENIZER_SHA, 'adapter_path': str(ADAPTER), 'adapter_weights_sha256': ADAPTER_SHA,
            'adapter_selection': json_file(files['adapter_best']), 'legacy_system_sha256': system_sha,
            'historical_prompt_function_sha256': prompt_function_fingerprint(current),
            'legacy_zero_shot_max_length': base_run['max_length'], 'legacy_icl_max_length': icl_run['max_length'],
            'generation': 'greedy, max_new_tokens=128, thinking disabled',
            'reconstruction_note': 'Legacy zero-shot messages/rendering reconstructed from matching saved source '
                                   'and current cached tokenizer; those prompts were not captured at generation time.'}


def context_row(cohort: Cohort, record: dict[str, Any]) -> dict[str, Any]:
    source_family = record.get('evaluation_family') or record['family']
    boxes = FAMILY_COUNTS[source_family]
    board = record['start_observation']
    return {'case_id': f'{cohort}:{record["record_id"]}:identity',
            'cohort_id': cohort, 'context_id': record['record_id'], 'historical_record_id': record['record_id'],
            'layout_id': record['layout_id'], 'family': f'12x12x{boxes}', 'source_family': source_family,
            'split': 'test', 'transform': 'identity', 'source_rows': board, 'board': board,
            'task_goal': record['task_goal'], 'source_path': record['source_path'],
            'source_sha256': record.get('source_sha256'), 'trajectory_id': record.get('trajectory_id'),
            'expected_methods': list(COHORT_METHODS[cohort]), 'historical_exposure': 'previously_evaluated'}


def build(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite a frozen stage03 dataset')
    files = source_files()
    protocol = ensure_protocol(files)
    legacy = read_selected_data(files['transfer_data'], files['transfer_ids'], 40)
    representation = read_selected_data(files['representation_data'], files['representation_ids'], 128)
    original = by_record(jsonl(files['original_data']), 'original data')
    raw = {name: by_record(jsonl(files[name]), name) for name in
           ('transfer_base', 'transfer_adapter', 'legacy_icl', 'representation_base', 'representation_icl')}
    prompts = {name: by_record(jsonl(files[name]), name) for name in
               ('legacy_icl_prompts', 'representation_base_prompts', 'representation_icl_prompts')}
    legacy_ids = {row['record_id'] for row in legacy}
    representation_ids = {row['record_id'] for row in representation}
    if set(raw['transfer_base']) != legacy_ids or set(raw['transfer_adapter']) != legacy_ids:
        raise ValueError('Saved zero-shot outputs do not cover the frozen transfer cohort')
    if any(set(raw[f'representation_{name}']) != representation_ids
           or set(prompts[f'representation_{name}_prompts']) != representation_ids for name in ('base', 'icl')):
        raise ValueError('Saved representation outputs or prompts do not cover the frozen cohort')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    contexts: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for cohort, records in ((Cohort.LEGACY, legacy), (Cohort.REPRESENTATION, representation)):
        for record in records:
            context = context_row(cohort, record)
            contexts.append(context)
            identifier = record['record_id']
            for method in COHORT_METHODS[cohort]:
                prediction = None
                prompt: dict[str, Any] | None = None
                output_file: Path | None = None
                prompt_file: Path | None = None
                prompt_origin = None
                if cohort == Cohort.LEGACY and method in (Method.BASE, Method.ADAPTER):
                    name = 'transfer_base' if method == Method.BASE else 'transfer_adapter'
                    prediction, output_file = raw[name][identifier], files[name]
                    messages = prompt_messages(record)
                    prompt = {'messages': messages, 'rendered_prompt': tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)}
                    prompt_origin = 'reconstructed_from_historical_source_and_cached_tokenizer'
                elif cohort == Cohort.LEGACY:
                    if context['family'] == '12x12x10':
                        if (identifier not in original or identifier not in raw['legacy_icl']
                                or identifier not in prompts['legacy_icl_prompts']
                                or original[identifier]['start_observation'] != context['board']
                                or original[identifier]['task_goal'] != context['task_goal']):
                            raise ValueError('Legacy ICL input does not match the transferred context')
                        prediction, output_file = raw['legacy_icl'][identifier], files['legacy_icl']
                        prompt, prompt_file = prompts['legacy_icl_prompts'][identifier], files['legacy_icl_prompts']
                        current_messages = prompt_messages(record)
                        if (prompt['messages'][0] != current_messages[0]
                                or prompt['messages'][-1] != current_messages[-1]):
                            raise ValueError('Saved ICL query/system differ from the legacy zero-shot protocol')
                        prompt_origin = 'captured_historical_prompt'
                else:
                    name = 'representation_base' if method == Method.BASE else 'representation_icl'
                    prediction, output_file = raw[name][identifier], files[name]
                    prompt_file = files[f'{name}_prompts']
                    prompt = prompts[f'{name}_prompts'][identifier]
                    prompt_origin = 'captured_historical_prompt'
                if prediction is not None:
                    if (prediction['layout_id'] != context['layout_id'] or prediction['split'] != 'test'
                            or not isinstance(prediction.get('raw_output'), str)):
                        raise ValueError('A historical prediction does not match its frozen context')
                    if prompt is None:
                        raise ValueError('An available prediction has no prompt provenance')
                outputs.append({**context, 'method': method,
                                'prediction_status': PredictionStatus.COMPLETE if prediction else PredictionStatus.MISSING,
                                'missing_reason': None if prediction else 'not_collected_in_historical_protocol',
                                'raw_output': prediction['raw_output'] if prediction else None,
                                'historical_prediction': prediction, 'historical_output_path': str(output_file) if output_file else None,
                                'historical_output_sha256': sha(output_file) if output_file else None,
                                'prompt': prompt, 'prompt_origin': prompt_origin,
                                'prompt_source_path': str(prompt_file) if prompt_file else None,
                                'prompt_source_sha256': sha(prompt_file) if prompt_file else None})
    output.mkdir(parents=True, exist_ok=True)
    stage1 = args.stage01_data.resolve()
    old_exposure = json_file(stage1 / 'exposure_register.json')
    train_path = args.stage02_train.resolve() / 'exposure_register.json'
    train_exposure = json_file(train_path)
    layouts = {row['layout_id'] for row in contexts}
    train_layouts = set(train_exposure['train_layout_ids'])
    if layouts & train_layouts:
        raise ValueError('Historical evaluation overlaps the newly assigned campaign training cohort')
    exposure = {'schema_version': 1, 'legacy_and_representation_overlap_layouts': sorted(
                    {row['layout_id'] for row in contexts if row['cohort_id'] == Cohort.LEGACY}
                    & {row['layout_id'] for row in contexts if row['cohort_id'] == Cohort.REPRESENTATION}),
                'stage01_evaluation_layouts_in_history': sorted(layouts & set(old_exposure['campaign_evaluation_layout_ids'])),
                'stage01_development_layouts_in_history': sorted(layouts & set(old_exposure['development_layout_ids'])),
                'stage02_train_overlap_layouts': [], 'all_historical_contexts': 'previously_evaluated',
                'old_checkpoint_training_exposure': 'unknown',
                'stage01_exposure_path': str(stage1 / 'exposure_register.json'),
                'stage01_exposure_sha256': sha(stage1 / 'exposure_register.json'),
                'stage02_train_exposure_path': str(train_path), 'stage02_train_exposure_sha256': sha(train_path)}
    write_jsonl(output / 'contexts.jsonl', contexts)
    write_jsonl(output / 'outputs.jsonl', outputs)
    write_json(output / 'exposure_register.json', exposure)
    counts = Counter((str(row['cohort_id']), str(row['method']), row['family'], str(row['prediction_status']))
                     for row in outputs)
    manifest = {'schema_version': 1, 'cohort_methods': COHORT_METHODS, 'protocol': protocol,
                'source_files': {name: {'path': str(path), 'sha256': sha(path)} for name, path in files.items()},
                'selection': 'Use pre-existing frozen record order and one context per D4 layout; '
                             'no output or success-based selection.',
                'cohort_counts': {str(Cohort.LEGACY): len(legacy), str(Cohort.REPRESENTATION): len(representation)},
                'prediction_counts': [{'cohort_id': a, 'method': b, 'family': c, 'status': d, 'count': n}
                                      for (a, b, c, d), n in sorted(counts.items())],
                'limitations': ['Historical test contexts were already inspected.',
                                'The representation and legacy cohorts use different prompt protocols.',
                                'Legacy ICL was not collected on the 20 x4 contexts.',
                                'Family and source distribution vary together; no isolated box-count comparison.',
                                'The historical zero-shot prompt rendering is reconstructed, not captured.',
                                'Legacy metadata records the model revision and path but not a weight hash '
                                'at inference time; the current model hash does not establish its past contents.'],
                'output_sha256': {name: sha(output / name) for name in
                                  ('contexts.jsonl', 'outputs.jsonl', 'exposure_register.json')},
                'preparer_sha256': sha(Path(__file__))}
    write_json(output / 'data_manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage01-data', type=Path, required=True)
    parser.add_argument('--stage02-train', type=Path, required=True)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({'cohort_counts': result['cohort_counts'], 'prediction_counts': result['prediction_counts']}))


if __name__ == '__main__':
    main()
