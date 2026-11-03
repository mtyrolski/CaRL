"""Run a tiny offline search with existing trained Sokoban checkpoints.

This is an engineering probe, not a controlled comparison with Luna. The
checkpoints use separate fixed-offset generators trained on 10x10 boards.
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
import torch
from transformers import BartForConditionalGeneration, BertForSequenceClassification

from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.environment.training_goal import TrainingGoal
from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash, validate_subgoal
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import reverse_board
from carl.icl.search import search
from carl.icl.search_tree import SearchTreeRecorder
from carl.icl.types import Board, Probabilities
from scripts.experiment_support.trace import verify_tree


class ProbeConfig(TypedDict):
    """Input settings for a single small, offline fixture."""

    checkpoint_root: str
    generator_scales: list[int]
    controller_scale: int
    board_size: int
    boxes: int
    board_seed: int
    search_seed: int
    action_budget: int
    max_expansions: int
    num_subgoals: int
    torch_threads: int
    output: str
    num_beams: NotRequired[int]
    weights_file: NotRequired[str]
    remove_border: NotRequired[bool]
    generator_paths: NotRequired[dict[str, str]]
    controller_path: NotRequired[str]
    value_path: NotRequired[str]
    validation_manifest: NotRequired[str]
    validation_index: NotRequired[int]


def checkpoint(root: Path, role: str, weights_file: str) -> Path:
    """Require local weights; never fall back to network or remote code."""
    path = root / role
    for name in ('config.json', weights_file):
        if not (path / name).is_file():
            raise FileNotFoundError(f'Missing {name}: {path}')
    return path


def digest(path: Path) -> str:
    """Fingerprint the weights without loading the optimizer state."""
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


class TrainedComponents:
    """Adapt trained explicit-state CaRL models to the common action-budget search."""

    def __init__(self, config: ProbeConfig) -> None:
        root = Path(config['checkpoint_root']).expanduser().resolve()
        remove_border = config.get('remove_border', True)
        token_size = config['board_size'] - 2 if remove_border else config['board_size']
        self.tokenizer = SokobanTokenizer(size_of_board=(token_size, token_size), remove_border=remove_border,
                                          type_of_value_training='regression')
        self.env = SokobanEnv(self.tokenizer, num_boxes=config['boxes'])
        self.scales = config['generator_scales']
        self.n_subgoals = config['num_subgoals']
        self.num_beams = config.get('num_beams', self.n_subgoals)
        if self.num_beams < self.n_subgoals:
            raise ValueError('num_beams must be at least num_subgoals')
        self.diagnostics: Counter[str] = Counter()
        self.weights_file = config.get('weights_file', 'model.safetensors')
        if self.weights_file not in {'model.safetensors', 'pytorch_model.bin'}:
            raise ValueError('Unsupported local weights format')
        generator_paths = config.get('generator_paths', {})
        self.paths = {f'generator/{scale}': checkpoint(
            root, generator_paths.get(str(scale), f'generator/{scale}'), self.weights_file
        ) for scale in self.scales}
        self.paths['cllp'] = checkpoint(root, config.get('controller_path', f"cllp/{config['controller_scale']}"),
                                        self.weights_file)
        self.paths['value'] = checkpoint(root, config.get('value_path', 'value'), self.weights_file)
        use_safetensors = self.weights_file == 'model.safetensors'
        self.generators = {
            scale: BartForConditionalGeneration.from_pretrained(
                self.paths[f'generator/{scale}'], local_files_only=True, use_safetensors=use_safetensors
            ).eval() for scale in self.scales
        }
        self.policy = BertForSequenceClassification.from_pretrained(
            self.paths['cllp'], local_files_only=True, use_safetensors=use_safetensors
        ).eval()
        self.value_network = BertForSequenceClassification.from_pretrained(
            self.paths['value'], local_files_only=True, use_safetensors=use_safetensors
        ).eval()
        self.max_new_tokens = token_size * token_size + 1

    @torch.inference_mode()
    def propose(self, state: Board, scale: float) -> list[Board]:
        """Decode full boards and discard invalid, copied, or repeated proposals."""
        if scale != int(scale) or int(scale) not in self.generators:
            raise ValueError(f'No trained generator for scale {scale}')
        encoded, _ = self.tokenizer.x_y_tokenizer(state, state, TrainingGoal.GENERATOR)
        outputs = self.generators[int(scale)].generate(
            encoded, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams,
            num_return_sequences=self.n_subgoals, do_sample=False,
        )
        if not isinstance(outputs, torch.Tensor):
            raise TypeError('Expected generated token sequences as a tensor')
        proposals: list[Board] = []
        seen: set[bytes] = set()
        self.diagnostics['generator_calls'] += 1
        for output in outputs.tolist():
            decoded = self.tokenizer.board_detokenizer(output)
            if decoded is None or not isinstance(decoded, np.ndarray):
                self.diagnostics['invalid_proposals'] += 1
                continue
            proposal = np.asarray(decoded, dtype=np.float32)
            try:
                validate_subgoal(state, proposal)
            except ValueError:
                self.diagnostics['invalid_proposals'] += 1
                continue
            key = proposal.argmax(axis=-1).tobytes()
            if np.array_equal(state, proposal):
                self.diagnostics['self_copies'] += 1
            elif key in seen:
                self.diagnostics['duplicate_proposals'] += 1
            else:
                seen.add(key)
                proposals.append(proposal)
        return proposals

    @torch.inference_mode()
    def probabilities(self, state: Board, subgoal: Board) -> Probabilities:
        """Match CaRL's greedy controller, expressed as a categorical distribution."""
        encoded, _ = self.tokenizer.x_y_tokenizer((state, subgoal), 0, TrainingGoal.CLLP)
        action = int(self.policy(encoded).logits[0].argmax().item())
        distribution = np.zeros(4, dtype=np.float64)
        distribution[action] = 1.0
        self.diagnostics['policy_calls'] += 1
        return distribution

    @torch.inference_mode()
    def value(self, state: Board) -> float:
        """Use the same negated distance as CaRL's TransformerValue."""
        encoded, _ = self.tokenizer.x_y_tokenizer(state, 0, TrainingGoal.VALUE)
        self.diagnostics['value_calls'] += 1
        return -float(self.value_network(encoded).logits[0, 0].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/icl/trained_sokoban_torch_probe.json')
    parser.add_argument('--validation-index', type=int, help='Select a full board from the configured validation manifest')
    parser.add_argument('--output', type=Path, help='Write the probe result to this separate file')
    args = parser.parse_args()
    raw: dict[str, Any] = json.loads(args.config.read_text(encoding='utf-8'))
    if args.validation_index is not None:
        if args.validation_index < 0 or not raw.get('validation_manifest'):
            raise ValueError('A nonnegative index requires a configured validation manifest')
        raw['validation_index'] = args.validation_index
    if args.output is not None:
        raw['output'] = str(args.output)
    config = cast(ProbeConfig, raw)
    if (config['board_size'] not in (10, 12) or config['boxes'] != 4 or config['action_budget'] > 128
            or config['num_subgoals'] not in (1, 2)):
        raise ValueError('This probe requires 10x10x4 or 12x12x4, B<=128, and one or two proposals')
    if len(set(config['generator_scales'])) != len(config['generator_scales']):
        raise ValueError('Duplicate generator scales')
    torch.set_num_threads(config['torch_threads'])
    manifest_path = config.get('validation_manifest')
    if manifest_path:
        path = ROOT / manifest_path
        records = json.loads(path.read_text(encoding='utf-8'))['instances']
        instance = records[config.get('validation_index', 0)]
        initial = from_rows(instance['state'])
        metadata: dict[str, Any] = {'id': instance['id'], 'source': str(path.relative_to(ROOT)),
                                    'manifest_sha256': digest(path)}
    else:
        initial, _private_witness, generated = reverse_board(config['board_size'], config['boxes'], config['board_seed'])
        metadata = {'source': 'reverse_board-v1 synthetic fixture', 'generator': generated}
    if initial.shape != (config['board_size'], config['board_size'], 7):
        raise ValueError('Board dimensions do not match the configured model')
    components = TrainedComponents(config)
    search_config = ExperimentConfig.model_validate({
        'method': 'fixed', 'scales': config['generator_scales'], 'fixed_cap': 'horizon',
        'action_budget': config['action_budget'], 'max_expansions': config['max_expansions'],
        'calibration': {'action_cap': max(config['generator_scales'])},
    })
    recorder = SearchTreeRecorder()
    started = time.perf_counter()
    result = search(components.env, components, initial, search_config, config['search_seed'], observer=recorder)
    elapsed = time.perf_counter() - started
    verification = verify_tree(recorder, initial, config['action_budget'])
    payload = {
        'purpose': 'offline one-board engineering probe; no API and no benchmark claim',
        'model_family': 'pretrained fixed-offset Sokoban BART/BERT, not a relative-scale model',
        'board': {'size': config['board_size'], 'boxes': config['boxes'], 'seed': config['board_seed'],
                  'state_hash': state_hash(initial), **metadata},
        'controller': {'scale': config['controller_scale'], 'sampling': 'greedy argmax as CaRL BasicValidator'},
        'search': 'shared action-budget search with primitive fallback; not original CaRL node-budget AdaSubS',
        'checkpoint_sha256': {role: digest(path / components.weights_file) for role, path in components.paths.items()},
        'config': dict(config), 'outcome': asdict(result), 'verification': verification,
        'diagnostics': dict(components.diagnostics), 'seconds': elapsed,
    }
    output = Path(config['output']).expanduser()
    output = output if output.is_absolute() else ROOT / output
    write_json(output, payload)
    print(json.dumps({'output': str(output), 'solved': result.solved, 'finish_reason': result.finish_reason,
                      'actions': result.attempted_actions, 'verified': verification['verified'],
                      'diagnostics': dict(components.diagnostics)}, indent=2))


if __name__ == '__main__':
    main()
