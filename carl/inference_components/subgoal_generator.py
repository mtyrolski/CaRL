from abc import abstractmethod
from collections.abc import Callable
from typing import Any, cast

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
        generator: Callable[[str], PreTrainedModel] | type[nn.Module],
        path_to_generator_weights: str,
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int | bool | float] | None = None,
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
        generator_network_class: Callable[[str], PreTrainedModel] | type[PreTrainedModel],
        path_to_generator_weights: str,
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int | bool | float] | None,
    ) -> None:
        super().__init__(generator_network_class, path_to_generator_weights, env, subgoal_generation_kwargs)
        self.device: torch.device = torch.device('cpu')
        self.mode: str = "bank"
        self.subgoal_generation_kwargs = subgoal_generation_kwargs
        self.generator_network_class: Callable[[str], PreTrainedModel] | type[PreTrainedModel] = generator_network_class
        self.sub_generator: PreTrainedModel | None = None

    def construct_network(self) -> None:
        # We do not put the generator on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.sub_generator = cast(
            PreTrainedModel,
            self.instantiate_network(self.generator_network_class, self.path_to_generator_weights),
        )

    def get_network(self) -> PreTrainedModel:
        assert self.sub_generator is not None, "Subgoal generator network is not constructed."
        return self.sub_generator

    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        """
        Generate sub-goals for the given state.

        :param state: the state.
        :return: the subgoals.
        """
        state = node.state

        if self.subgoal_generation_kwargs is None:
            raise ValueError("subgoal_generation_kwargs must be provided.")
        if self.sub_generator is None:
            raise RuntimeError("Subgoal generator network is not constructed.")

        subgoal_generation_kwargs = self.subgoal_generation_kwargs
        max_new_tokens: int = int(subgoal_generation_kwargs["max_new_tokens"])
        num_beams: int = int(subgoal_generation_kwargs["num_beams"])
        num_return_sequences: int = int(subgoal_generation_kwargs["num_return_sequences"])
        do_sample: bool = bool(subgoal_generation_kwargs.get("do_sample", False))
        temperature: float = float(subgoal_generation_kwargs.get("temperature", 1.0))

        if do_sample:
            num_beams = 1

        encoded_board: torch.Tensor
        encoded_board, _ = self.env.tokenizer.x_y_tokenizer(x=state, y=state, training_goal=TrainingGoal.GENERATOR)
        encoded_board = encoded_board.to(self.device)
        with torch.no_grad():
            generate = cast(Callable[..., torch.Tensor], self.sub_generator.generate)
            outputs: list[list[int]] = generate(
                encoded_board,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature,
            ).tolist()
        subgoals: list[GeneratedSubgoal] = []

        for output in outputs:
            subgoal: np.ndarray | str | None = self.env.tokenizer.board_detokenizer(output)
            if subgoal is not None:
                subgoals.append(GeneratedSubgoal(subgoal, {}))

        return subgoals


class UniversalPropositionalSubgoalGenerator(TransformerSubgoalGenerator):
    """Universal generator that proposes candidate subgoal states without explicit distance labels.

    This component reuses the existing seq2seq generator I/O format (`state -> subgoal state`)
    but returns proposal metadata such as rank/confidence and declares
    `mode='propositional_universal'` to enable no-k solver/planner behavior.
    """

    def __init__(
        self,
        generator_network_class: Callable[[str], PreTrainedModel] | type[PreTrainedModel],
        path_to_generator_weights: str,
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int | bool | float] | None,
        proposal_confidence_mode: str = "rank_inverse",
        mode: str = "propositional_universal",
    ) -> None:
        super().__init__(
            generator_network_class=generator_network_class,
            path_to_generator_weights=path_to_generator_weights,
            env=env,
            subgoal_generation_kwargs=subgoal_generation_kwargs,
        )
        self.mode = mode
        # Empty list keeps backward-compatible solver code paths safe (no per-k metrics in universal mode).
        self.generator_k_list: list[int] = []
        self.proposal_confidence_mode = proposal_confidence_mode

    def _proposal_confidence(self, rank: int, output: list[int]) -> float:
        if self.proposal_confidence_mode == "rank_inverse":
            return 1.0 / float(rank + 1)
        if self.proposal_confidence_mode == "uniform":
            return 1.0
        # Fallback to a deterministic bounded score if a custom mode name is passed.
        return 1.0 / float(rank + 1)

    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        if self.subgoal_generation_kwargs is None:
            raise ValueError("subgoal_generation_kwargs must be provided.")
        if self.sub_generator is None:
            raise RuntimeError("Subgoal generator network is not constructed.")

        state = node.state
        subgoal_generation_kwargs = self.subgoal_generation_kwargs
        max_new_tokens: int = int(subgoal_generation_kwargs["max_new_tokens"])
        num_beams: int = int(subgoal_generation_kwargs["num_beams"])
        num_return_sequences: int = int(subgoal_generation_kwargs["num_return_sequences"])
        do_sample: bool = bool(subgoal_generation_kwargs.get("do_sample", False))
        temperature: float = float(subgoal_generation_kwargs.get("temperature", 1.0))

        if do_sample:
            num_beams = 1

        encoded_board, _ = self.env.tokenizer.x_y_tokenizer(x=state, y=state, training_goal=TrainingGoal.GENERATOR)
        encoded_board = encoded_board.to(self.device)
        with torch.no_grad():
            generate = cast(Callable[..., torch.Tensor], self.sub_generator.generate)
            outputs: list[list[int]] = generate(
                encoded_board,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature,
            ).tolist()

        subgoals: list[GeneratedSubgoal] = []
        seen_states: set[tuple[int, ...] | tuple[str, ...]] = set()
        for rank, output in enumerate(outputs):
            subgoal = self.env.tokenizer.board_detokenizer(output)
            if subgoal is None:
                continue

            if isinstance(subgoal, np.ndarray):
                key: tuple[int, ...] | tuple[str, ...] = tuple(map(int, subgoal.flatten()))
            else:
                key = tuple(subgoal)

            duplicate = key in seen_states
            if not duplicate:
                seen_states.add(key)

            metadata: dict[str, Any] = {
                "proposal_rank": rank,
                "proposal_confidence": self._proposal_confidence(rank, output),
                "proposal_duplicate_in_decode": duplicate,
                "generator_mode": self.mode,
            }
            if "depth" in node.metadata:
                metadata["node_depth"] = node.metadata["depth"]

            subgoals.append(GeneratedSubgoal(subgoal, metadata))

        return subgoals


class AdaptiveSubgoalGenerator(InferenceComponent):
    def __init__(
        self,
        generator_k_list: list[int],
        paths_to_generator_weights: list[str],
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int | bool | float] | None,
        subgoal_generator_class: Callable[..., SubgoalGenerator] = TransformerSubgoalGenerator,
        mode: str = "bank",
    ) -> None:
        self.env = env
        self.subgoal_generation_kwargs = subgoal_generation_kwargs
        self.generator_k_list = generator_k_list
        self.mode: str = mode

        self.subgoal_generators: dict[int, SubgoalGenerator] = {
            idx: subgoal_generator_class(
                env=env,
                subgoal_generation_kwargs=subgoal_generation_kwargs,
                path_to_generator_weights=path,
            )
            for idx, path in zip(generator_k_list, paths_to_generator_weights)
        }

    def construct_network(self) -> None:
        for subgoal_generator in self.subgoal_generators.values():
            subgoal_generator.construct_network()

    def get_network(self) -> dict[int, PreTrainedModel]:
        return {
            idx: cast(PreTrainedModel, subgoal_generator.get_network())
            for idx, subgoal_generator in self.subgoal_generators.items()
        }

    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        """
        Generate sub-goals for the given state.

        :param state: the state.
        :return: the subgoals.
        """
        if node.next_expand_with_k_generator is None:
            raise ValueError("Node does not specify which k-generator to use.")
        subgoal_generator = self.subgoal_generators[node.next_expand_with_k_generator]
        return subgoal_generator.get_subgoals(node)
