from abc import abstractmethod
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor
from torch import nn
from transformers import PreTrainedModel

from carl.utils.aliases import State
from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.inference_components.component import InferenceComponent
from carl.inference_components.component import RawComponent
from carl.inference_components.component import TrainingModule
from carl.utils.loggers import log_error_and_raise


class ConditionalLowLevelPolicy(InferenceComponent):
    @abstractmethod
    def __init__(
        self,
        conditional_low_level_policy_class: Callable[[str], nn.Module] | type[nn.Module],
        path_to_conditional_low_level_policy_weights: str,
        env: GameEnv,
    ) -> None:
        """
        Initialize the conditional low level policy network.

        params:
            conditional_low_level_policy: the conditional low level policy.
            env: the environment.
        """

        self.conditional_low_level_policy_class = conditional_low_level_policy_class
        self.path_to_conditional_low_level_policy_weights = (path_to_conditional_low_level_policy_weights)
        self.env = env

    @abstractmethod
    def construct_network(self) -> None:
        """Construct the networks."""

        raise NotImplementedError

    @abstractmethod
    def get_action(
        self,
        state: State,
        state_after_k: State,
    ) -> Tensor:
        """
        Get the action from state to state_after_k.

        params:
            state: the state.
            state_after_k: the state after k steps.
        return:
            the action.
        """

        raise NotImplementedError


class TransformerConditionalLowLevelPolicy(ConditionalLowLevelPolicy):
    def __init__(
        self,
        conditional_low_level_policy_class: Callable[[str], PreTrainedModel] | type[PreTrainedModel],
        path_to_conditional_low_level_policy_weights: str,
        env: GameEnv,
        training_module: TrainingModule | None = None,
    ) -> None:
        super().__init__(conditional_low_level_policy_class, path_to_conditional_low_level_policy_weights, env)
        self.device: torch.device = torch.device('cpu')

        self.cllp: PreTrainedModel | None = None
        self.training_module = training_module

    def get_component_training_module(self) -> TrainingModule | dict[str, TrainingModule] | None:
        return self.training_module

    def construct_network(self) -> None:
        # We do not put the cllp on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.cllp = cast(
            PreTrainedModel,
            self.instantiate_network(
                self.conditional_low_level_policy_class,
                self.path_to_conditional_low_level_policy_weights,
            ),
        )

    def get_network(self) -> RawComponent:
        if self.cllp is None:
            log_error_and_raise(
                'Conditional low level policy network is not constructed. '
                'Call `construct_network()` before calling `get_network()`.'
            )
            
        assert isinstance(self.cllp, PreTrainedModel), (
            f'Expected cllp to be an instance of PreTrainedModel, got {type(self.cllp)}'
        )
            
        return self.cllp

    def get_action(self, state: State, state_after_k: State) -> Tensor:
        assert self.cllp is not None
        encoded_boards = torch.cat([
            self.env.tokenizer.x_y_tokenizer(x=(state, subgoal), y=0, training_goal=TrainingGoal.CLLP)[0]
            for state, subgoal in zip([state], [state_after_k])
        ])
        encoded_boards = encoded_boards.to(self.device)

        with torch.no_grad():
            logits: torch.Tensor = self.cllp(encoded_boards).logits    # (BS, NUM_ACTIONS)
        assert logits.shape[0] == 1, f'Expected batch size 1, got {logits.shape[0]}'

        return logits[0]    # Distriubtion over actions
