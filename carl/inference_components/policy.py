from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor
from torch import nn
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.inference_components.subgoal_generator import SubgoalGenerator
from carl.inference_components.component import InferenceComponent
from carl.solver.nodes import GeneratedAction
from carl.solver.nodes import GeneratedSubgoal
from carl.solver.nodes import SearchTreeNode


class Policy(InferenceComponent):
    @abstractmethod
    def __init__(
        self,
        policy_network_class: type[nn.Module] | None,
        env: GameEnv,
    ) -> None:
        """
        Initialize the policy network.

        params:
            policy_network: the policy network.
            env: the environment.
        """

        self.policy_network_class = policy_network_class
        self.env = env

    @abstractmethod
    def construct_network(self) -> None:
        """Construct the networks."""

        raise NotImplementedError

    @abstractmethod
    def get_actions(self, state: np.ndarray | str) -> Tensor:
        """
        Get a list of actions for the given state.

        params:
            state: the state.
        return:
            the actions.
        """

        raise NotImplementedError


class TransformerPolicy(Policy):
    def __init__(
        self,
        policy_network_class: type[PreTrainedModel],
        env: GameEnv,
        path_to_policy_weights: str,
        n_actions: int | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        super().__init__(policy_network_class, env)
        print(torch.cuda.is_available())
        self.device: torch.device = torch.device('cpu')
        print(f'Using device: {self.device}')
        self.path_to_policy_weights = path_to_policy_weights
        self.n_actions = n_actions
        self.confidence_threshold = confidence_threshold

        assert (n_actions is not None) or (confidence_threshold
                                           is not None), 'Either n_actions or confidence_threshold must be specified.'

        self.policy: PreTrainedModel | None = None

    def construct_network(self) -> None:
        # We do not put the policy on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.policy = self.instantiate_network(self.policy_network_class, self.path_to_policy_weights)

    def get_network(self) -> nn.Module:
        return self.policy

    def get_actions(self, state: np.ndarray | str) -> list[GeneratedAction]:
        encoded_board: torch.Tensor
        encoded_board, _ = self.env.tokenizer.x_y_tokenizer(x=state, y=0, training_goal=TrainingGoal.POLICY)
        encoded_board = encoded_board.to(self.device)

        with torch.no_grad():
            output: Tensor = self.policy(encoded_board).logits

        output = output.squeeze(dim=0)

        if self.confidence_threshold is not None:
            actions_to_return: list[GeneratedAction] = []
            cumulative_prob: float = 0.0
            action_with_prob: list[tuple[int, float]] = list(
                zip(range(len(output)),
                    torch.softmax(output, dim=-1).cpu().numpy()))
            action_with_prob.sort(key=lambda x: x[1], reverse=True)

            for action, prob in action_with_prob:
                cumulative_prob += prob
                actions_to_return.append(GeneratedAction(action, {'action_probs': output}))
                if cumulative_prob >= self.confidence_threshold:
                    break
            return actions_to_return

        actions_probs = list(zip(range(len(output)), torch.softmax(output, dim=-1).cpu().numpy()))
        actions_probs_sorted_form_highest_probability = sorted(actions_probs, key=lambda x: x[1], reverse=True)
        taking_n_actions = actions_probs_sorted_form_highest_probability[:self.n_actions]
        return taking_n_actions


class TransformerPolicyGeneration(Policy):
    def __init__(
        self,
        policy_network_class: type[PreTrainedModel],
        env: GameEnv,
        path_to_policy_weights: str,
        subgoal_generation_kwargs: dict[str, int] | None,
    ) -> None:
        super().__init__(policy_network_class, env)
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path_to_policy_weights = path_to_policy_weights
        self.subgoal_generation_kwargs = subgoal_generation_kwargs

        self.policy: PreTrainedModel | None = None

    def construct_network(self) -> None:
        self.policy = self.instantiate_network(self.policy_network_class, self.path_to_policy_weights)

    def get_network(self) -> nn.Module:
        return self.policy

    def get_actions(self, state: np.ndarray | str) -> set[GeneratedAction]:
        max_new_tokens: int = self.subgoal_generation_kwargs['max_new_tokens']
        num_beams: int = self.subgoal_generation_kwargs['num_beams']
        num_return_sequences: int = self.subgoal_generation_kwargs['num_return_sequences']

        encoded_board: torch.Tensor
        encoded_board, _ = self.env.tokenizer.x_y_tokenizer(x=state, y=0, training_goal=TrainingGoal.POLICY_GENERATION)
        encoded_board = encoded_board.to(self.device)
        with torch.no_grad():
            outputs: list[list[int]] = self.policy.generate(
                encoded_board,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            ).tolist()

        moves: set[GeneratedAction] = set()

        for output in outputs:
            move: int | None = self.env.tokenizer.action_detokenizer(output)
            if move is not None:
                moves.add(GeneratedAction(move, {}))

        return moves


class ExhaustiveBaselinePolicy(Policy):
    def __init__(
        self,
        n_actions: int,
    ) -> None:
        super().__init__(None, None)
        self.n_actions: int = n_actions

    def construct_network(self) -> None:
        pass

    def get_actions(self, state: np.ndarray):
        return [GeneratedAction(action, {}) for action in range(self.n_actions)]


class PolicyGeneratorWrapper(SubgoalGenerator):
    def __init__(
        self,
        policy: Policy,
        env: GameEnv,
    ) -> None:
        super().__init__(None, '', env, None)
        self.policy = policy
        self.generator_k_list = []

    def construct_network(self) -> None:
        self.policy.construct_network()

    def get_network(self) -> nn.Module:
        return self.policy.policy_network

    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        state = node.state

        actions = self.policy.get_actions(state)

        for action in actions:
            action.generation_metadata['action'] = action.action

        return [
            GeneratedSubgoal(self.env.next_state(state, action.action), action.generation_metadata)
            for action in actions
        ]
