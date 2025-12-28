import numpy as np
import torch
import pytest
from torch import Tensor

from carl.inference_components.value import (
    TransformerValue,
    TransformerValueGeneration,
)
from carl.environment.env import GameEnv

# Dummy environment and tokenizer to simulate inference
class DummyTokenizer:
    def x_y_tokenizer(self, x, y, training_goal):
        # return a simple tensor and dummy y
        return torch.tensor([[1.0, 2.0, 3.0]]), None

class DummyEnv(GameEnv):
    def __init__(self):
        # Initialize tokenizer
        self._tokenizer = DummyTokenizer()
    @property
    def tokenizer(self) -> DummyTokenizer:
        """Tokenizer property satisfying abstract base class."""
        return self._tokenizer

    @property
    def name(self) -> str:
        return 'dummy'
    def detect_action(self, board_before, board_after):
        return 0
    @staticmethod
    def distribution_to_action(distribution: Tensor) -> int:
        return 0
    def step(self, action: int):
        # not used
        return None, 0.0, False, {}
    def next_state(self, state, action: int):
        return state
    def is_solved(self, board):
        return False
    def state_to_repr(self, state, title=None):
        return str(state)
    def many_states_to_repr(self, states, title=None):
        return states
    def set_state(self, state) -> None:
        pass

@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(0)
    np.random.seed(0)

class DummyModelClassification:
    def __call__(self, x: Tensor):
        class Out:
            logits = torch.tensor([[0.0, 1.0, 2.0]])
        return Out()

class DummyModelRegression:
    def __call__(self, x: Tensor):
        class Out:
            logits = torch.tensor([[5.0]])
        return Out()

class DummyGenModel:
    @classmethod
    def from_pretrained(cls, path):
        return cls()
    def to(self, device):
        pass
    def generate(self, x, max_new_tokens, num_beams, num_return_sequences):
        # return tensor (num_return_sequences, length)
        return torch.tensor([[0.0, 1.0, 2.0, 3.0]])

def test_transformer_value_classification(monkeypatch):
    env = DummyEnv()
    tv = TransformerValue(
        value_network_class=DummyModelClassification,
        path_to_value_network_weights='unused',
        env=env,
        type_of_evaluation='classification',
        noise_variance=0.0,
    )
    # patch instantiate_network
    monkeypatch.setattr(TransformerValue, 'instantiate_network', lambda self, cls, path: DummyModelClassification())
    tv.construct_network()
    val = tv.get_value(state=np.zeros((1,)))
    # compute expected softmax
    dist = torch.softmax(torch.tensor([0.0,1.0,2.0]), dim=0)
    exp_value = (dist * torch.arange(3, dtype=torch.float)).sum().item()
    assert pytest.approx(val, rel=1e-5) == -exp_value

def test_transformer_value_regression(monkeypatch):
    env = DummyEnv()
    tv = TransformerValue(
        value_network_class=DummyModelRegression,
        path_to_value_network_weights='unused',
        env=env,
        type_of_evaluation='regression',
        noise_variance=0.0,
    )
    monkeypatch.setattr(TransformerValue, 'instantiate_network', lambda self, cls, path: DummyModelRegression())
    tv.construct_network()
    val = tv.get_value(state=np.zeros((1,)))
    assert val == -5.0

def test_transformer_value_generation(monkeypatch):
    env = DummyEnv()
    tg = TransformerValueGeneration(
        value_network=DummyGenModel,
        path_to_value_network_weights='unused',
        env=env,
        type_of_evaluation='generation',
        value_generation_kwargs={'max_new_tokens': 2, 'num_beams': 1, 'num_return_sequences': 1},
    )
    # patch from_pretrained
    monkeypatch.setattr(DummyGenModel, 'from_pretrained', classmethod(lambda cls, path: DummyGenModel()))
    tg.construct_network()
    result = tg.get_value(state=np.zeros((1,)))
    # logits [0,1,2,3]
    dist = torch.softmax(torch.tensor([[0.0,1.0,2.0,3.0]]), dim=1)
    exp = (dist * torch.arange(4, dtype=torch.float)).sum().item()
    assert pytest.approx(result, rel=1e-5) == exp
