from abc import abstractmethod

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.inference_components.component import InferenceComponent
from carl.solver.nodes import GeneratedSubgoal
from carl.solver.nodes import SearchTreeNode


class SubgoalGenerator(InferenceComponent):
    @abstractmethod
    def __init__(
        self,
        generator: type[nn.Module],
        path_to_generator_weights: str,
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int] | None = None,
    ) -> None:
        """
        Initialize the sub-goals generator.

        params:
            generator: the generator.
            env: the environment.
            subgoal_generation_kwargs: the subgoal generation kwargs.
        """

        self.generator = generator
        self.path_to_generator_weights = path_to_generator_weights
        self.env = env
        self.subgoal_generation_kwargs = subgoal_generation_kwargs

    @abstractmethod
    def construct_network(self) -> None:
        """Construct the networks."""

        raise NotImplementedError

    @abstractmethod
    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        """
        Generate sub-goals for the given state.

        params:
            state: the state.
        return:
            the subgoals.
        """

        raise NotImplementedError


class TransformerSubgoalGenerator(SubgoalGenerator):
    def __init__(
        self,
        generator_network_class: type[PreTrainedModel],
        path_to_generator_weights: str,
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int | bool] | None,
    ) -> None:
        super().__init__(generator_network_class, path_to_generator_weights, env, subgoal_generation_kwargs)
        self.device: torch.device = torch.device('cpu')
        self.subgoal_generation_kwargs = subgoal_generation_kwargs
        self.sub_generator: PreTrainedModel | None = None

    def construct_network(self) -> None:
        # We do not put the generator on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.sub_generator = self.instantiate_network(self.generator, self.path_to_generator_weights)

    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        return self.sub_generator

    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        """
        Generate sub-goals for the given state.

        :param state: the state.
        :return: the subgoals.
        """
        state = node.state

        max_new_tokens: int = self.subgoal_generation_kwargs['max_new_tokens']
        num_beams: int = self.subgoal_generation_kwargs['num_beams']
        num_return_sequences: int = self.subgoal_generation_kwargs['num_return_sequences']
        do_sample: bool = self.subgoal_generation_kwargs.get('do_sample', False)
        temperature: float = self.subgoal_generation_kwargs.get('temperature', 1.0)

        if do_sample:
            num_beams = 1

        encoded_board: torch.Tensor
        encoded_board, _ = self.env.tokenizer.x_y_tokenizer(x=state, y=state, training_goal=TrainingGoal.GENERATOR)
        encoded_board = encoded_board.to(self.device)
        with torch.no_grad():
            outputs: list[list[int]] = self.sub_generator.generate(
                encoded_board,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature,
            ).tolist()

        subgoals: list[GeneratedSubgoal] = []

        for output in outputs:
            subgoal: np.ndarray | None = self.env.tokenizer.board_detokenizer(output)
            if subgoal is not None:
                subgoals.append(GeneratedSubgoal(subgoal, {}))

        return subgoals


class AdaptiveSubgoalGenerator(InferenceComponent):
    def __init__(
        self,
        generator_k_list: list[int],
        paths_to_generator_weights: list[str],
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int] | None,
        subgoal_generator_class: type[SubgoalGenerator] = TransformerSubgoalGenerator,
    ) -> None:
        self.env = env
        self.subgoal_generation_kwargs = subgoal_generation_kwargs
        self.generator_k_list = generator_k_list

        self.subgoal_generators = {
            idx:
                subgoal_generator_class(env=env,
                                        subgoal_generation_kwargs=subgoal_generation_kwargs,
                                        path_to_generator_weights=path)
            for idx, path in zip(generator_k_list, paths_to_generator_weights)
        }

    def construct_network(self) -> None:
        for subgoal_generator in self.subgoal_generators.values():
            subgoal_generator.construct_network()

    def get_network(self) -> dict[str, PreTrainedModel]:
        return {idx: subgoal_generator.get_network() for idx, subgoal_generator in self.subgoal_generators.items()}

    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        """
        Generate sub-goals for the given state.

        :param state: the state.
        :return: the subgoals.
        """
        subgoal_generator = self.subgoal_generators[node.next_expand_with_k_generator]
        return subgoal_generator.get_subgoals(node)
