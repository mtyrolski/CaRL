"""Generator inputs exclude later boards and retain invalid proposals."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.environment.training_goal import TrainingGoal
from carl.icl.boards import canonical_layout_id
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts import generate_overnight_cpu_stage02 as generator


def board(boxes):
    tiles = np.ones((12, 12), dtype=int)
    tiles[[0, -1], :] = 0
    tiles[:, [0, -1]] = 0
    for col in range(2, boxes + 1):
        tiles[2 if col < 10 else 3, col if col < 10 else 2] = 3
    tiles[5, 3], tiles[5, 4], tiles[5, 5] = 5, 4, 2
    return np.eye(7, dtype=np.float32)[tiles]


def row(boxes):
    current = board(boxes)
    future = np.asarray(environment(current).next_state(current, 3))
    return {'context_id': f'context-{boxes}', 'layout_id': canonical_layout_id(current),
            'family': f'12x12x{boxes}', 'source_family': 'fixture', 'split': 'evaluation',
            'board_rank': 0, 'context_rank': 0, 'source': to_rows(current), 'source_hash': state_hash(current),
            'target': to_rows(future), 'target_hash': state_hash(future), 'k': 8, 'demonstration_actions': [3]}


def test_context_selection_strips_future_and_checks_identity():
    rows = [row(4), row(10)]
    selected = generator.select_contexts(rows, 'evaluation', 1)
    assert len(selected) == 2
    assert all(not {'target', 'target_hash', 'k', 'demonstration_actions'} & item.keys() for item in selected)
    rows[0]['source_hash'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        generator.select_contexts(rows, 'evaluation', 1)


def test_invalid_and_already_satisfied_outputs_are_retained():
    tokenizer = SokobanTokenizer()
    source = board(4)
    tokens = tokenizer.board_tokenizer(source).tolist()
    same = generator.decode_output(tokenizer, source, [0, *tokens, 2], 0)
    assert same['raw_token_ids'] and same['target'] is not None
    assert not same['valid_nontrivial'] and same['validation_error'] == 'already satisfied'
    invalid = generator.decode_output(tokenizer, source, [0, 15, 2], 1)
    assert invalid['raw_token_ids'] == [0, 15, 2]
    assert not invalid['valid_nontrivial'] and invalid['target'] is None


def test_model_receives_only_current_boards_and_saves_both_outputs(tmp_path, monkeypatch):
    rows = [row(4), row(10)]
    pairs = tmp_path / 'pairs.jsonl'
    pairs.write_text(''.join(json.dumps(item) + '\n' for item in rows))
    model_path = tmp_path / 'model'
    model_path.mkdir()
    for name in ('pytorch_model.bin', 'config.json', 'generation_config.json'):
        (model_path / name).write_bytes(name.encode())
    seen = []
    tokenizer = SokobanTokenizer()

    class FakeModel:
        def eval(self):
            return self

        def requires_grad_(self, _value):
            return self

        def generate(self, inputs, **kwargs):
            seen.append(inputs.clone())
            assert kwargs['num_return_sequences'] == 2 and not kwargs['do_sample']
            outputs = []
            for item in rows:
                target = environment(board(int(item['family'].split('x')[-1]))).next_state(
                    board(int(item['family'].split('x')[-1])), 3)
                sequence = [0, *tokenizer.board_tokenizer(np.asarray(target)).tolist(), 2]
                outputs.extend([sequence, sequence])
            return torch.tensor(outputs)

    monkeypatch.setattr(generator, 'MODEL', model_path)
    monkeypatch.setattr(generator.BartForConditionalGeneration, 'from_pretrained', lambda *a, **k: FakeModel())
    args = SimpleNamespace(pairs=pairs, output=tmp_path / 'output', split='evaluation', max_boards=1,
                           batch_size=2, threads=1, resume=False, stop_epoch=time.time() + 60,
                           stop_monotonic=time.monotonic() + 60)
    result = generator.run(args)
    assert result['state'] == 'complete' and result['completed_contexts'] == 2
    expected = torch.cat([tokenizer.x_y_tokenizer(board(boxes), board(boxes), TrainingGoal.GENERATOR)[0]
                          for boxes in (4, 10)])
    assert torch.equal(seen[0], expected)
    for item in rows:
        saved = json.loads((args.output / 'shards' / f"{item['context_id']}.json").read_text())
        assert len(saved['outputs']) == 2 and all(output['raw_token_ids'] for output in saved['outputs'])
    with pytest.raises(ValueError, match='--resume'):
        generator.run(args)
