from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor
from torch import nn
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.inference_components.component import InferenceComponent
from carl.inference_components.component import TrainingModule


class Value(InferenceComponent):
    @abstractmethod
    def __init__(
        self,
        value_network_class: type[nn.Module],
        path_to_value_network_weights: str,
        env: GameEnv,
        type_of_evaluation: str | None = None,
    ) -> None:
        """
        Initialize the value network.

        params:
            value_network: the value network.
            env: the environment.
        """

        self.value_network_class = value_network_class
        self.path_to_value_network_weights = path_to_value_network_weights
        self.env = env
        self.type_of_evaluation = type_of_evaluation

    @abstractmethod
    def construct_network(self) -> None:
        """Construct the networks."""

        raise NotImplementedError

    @abstractmethod
    def get_value(self, state: np.ndarray | str) -> float:
        """
        Get the value of the given state.

        params:
            state: the state.
        return:
            the value, for example, distance to the goal.
        """

        raise NotImplementedError


class TransformerValue(Value):
    def __init__(
        self,
        value_network_class: type[PreTrainedModel],
        path_to_value_network_weights: str,
        env: GameEnv,
        type_of_evaluation: str = 'classification',
        noise_variance: float = 0.0,
        training_module: TrainingModule | None = None,
    ) -> None:
        """
        Type of evaluation can be either 'classification', 'regression'.

        It corresponds to the type of the head of
        the model (type of training).
        """

        super().__init__(value_network_class, path_to_value_network_weights, env, type_of_evaluation)
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_variance = noise_variance

        self.value_network: PreTrainedModel | None = None
        self.training_module = training_module

        assert self.type_of_evaluation in ['regression', 'classification']

    def get_component_training_module(self) -> TrainingModule | dict[str, TrainingModule] | None:
        return self.training_module

    def construct_network(self) -> None:
        # We do not put the value on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.value_network = self.instantiate_network(self.value_network_class, self.path_to_value_network_weights)

    def get_value(self, state: np.ndarray | str) -> float:
        encoded_board: torch.Tensor
        encoded_board, _ = self.env.tokenizer.x_y_tokenizer(x=state, y=0, training_goal=TrainingGoal.VALUE)
        encoded_board = encoded_board.to(self.device)

        with torch.no_grad():
            output: Tensor = self.value_network(encoded_board).logits    # (BS, 1)

        if self.type_of_evaluation == 'classification':
            distance: float = self.expected_value(output)
        else:
            distance = output.flatten().tolist()[0]

        noisy_distance: float = -distance + np.random.normal(0, self.noise_variance)

        return noisy_distance

    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        return self.value_network

    def expected_value(self, logits) -> float:
        """Compute the expected value of the outputs, ensuring device consistency."""
        # logits: Tensor of shape (batch_size, num_classes)
        distribution: Tensor = torch.softmax(logits, dim=1)
        # Use the distribution's device for index tensor
        idx = torch.arange(distribution.shape[1], device=distribution.device)
        return (distribution * idx).sum().item()


class TransformerValueGeneration(Value):
    def __init__(
        self,
        value_network: type[PreTrainedModel],
        path_to_value_network_weights: str,
        env: GameEnv,
        type_of_evaluation: str = 'generation',
        value_generation_kwargs: dict[str, int] = None,
    ) -> None:
        super().__init__(value_network, path_to_value_network_weights, env, type_of_evaluation)
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.value: PreTrainedModel | None = None

        self.value_generation_kwargs = value_generation_kwargs

    def construct_network(self) -> None:
        # We do not put the value on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        # Instantiate the generation model from pretrained weights
        self.value = self.value_network_class.from_pretrained(self.path_to_value_network_weights)
        self.value.to(self.device)

    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        return self.value

    def get_value(self, state: np.ndarray | str) -> float:
        max_new_tokens: int = self.value_generation_kwargs['max_new_tokens']
        num_beams: int = self.value_generation_kwargs['num_beams']
        num_return_sequences: int = self.value_generation_kwargs['num_return_sequences']

        encoded_board: torch.Tensor
        encoded_board, _ = self.env.tokenizer.x_y_tokenizer(x=state, y=0, training_goal=TrainingGoal.VALUE_GENERATION)
        encoded_board = encoded_board.to(self.device)
        with torch.no_grad():
            outputs: list[list[int]] = self.value.generate(
                encoded_board,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            ).tolist()

        expected_value: float = self.expected_value(outputs[0])

        return expected_value

    def expected_value(self, logits) -> float:
        """Compute the expected value of the outputs for generation logits."""
        # Convert to tensor if logits is a list
        tensor: Tensor = logits if isinstance(logits, Tensor) else torch.tensor(logits, device=self.device, dtype=torch.float)
        # Apply softmax over the last dimension
        distribution: Tensor = torch.softmax(tensor, dim=-1)
        # Compute weighted sum of token indices
        idx = torch.arange(tensor.shape[-1], device=distribution.device)
        return (distribution * idx).sum().item()
