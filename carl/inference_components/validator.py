from abc import abstractmethod

import numpy as np
from torch import Tensor
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.inference_components.component import InferenceComponent
from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy
from carl.solver.nodes import ValidationResult


class Validator(InferenceComponent):
    """
    General Validator class.

    Used to check if a subgoal is achievable from a given state.
    """
    @abstractmethod
    def __init__(
        self,
        env: GameEnv,
        cllp: ConditionalLowLevelPolicy,
    ) -> None:
        self.cllp = cllp
        self.env = env

    @abstractmethod
    def construct_network(self) -> None:
        """Constructs the networks of the validator."""

        raise NotImplementedError()

    @abstractmethod
    def is_valid(self, state: np.ndarray | str, subgoal: np.ndarray | str) -> ValidationResult:
        """
        Checks if a subgoal is achievable from a given state.

        param state: the state.
        param subgoal: the subgoal.
        return: the validation result. The validation result is a tuple of the form (is_valid, is_solved, path, nodes_visited, achieved_state).

        is_valid: True if the subgoal is achievable from the given state, False otherwise.
        is_solved: True if the subgoal is achievable from the given state and the subgoal is a goal state, False otherwise.
        path: the path to the subgoal from the given state.
        nodes_visited: the number of nodes visited during the validation, i.e. the number of nodes that were expanded during the validation.
        achieved_state: the state that was achieved during the validation.
        """

        raise NotImplementedError()


class BasicValidator(Validator):
    """
    Validator class for TransformerConditionalLowLevelPolicy.

    To check if a subgoal is achievable from a given state,
    we use the conditional low level policy to generate a moves from the given state to the subgoal.
    We then use the environment to simulate the execution of the actions and check if the subgoal is achieved.
    """
    def __init__(self, env, cllp: ConditionalLowLevelPolicy, budget_for_achieving_subgoal) -> None:
        super().__init__(env, cllp)
        self.budget_for_achieving_subgoal = budget_for_achieving_subgoal

    def construct_network(self) -> None:
        self.cllp.construct_network()

    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        return self.cllp.get_network()

    def is_valid(self, state: np.ndarray | str, subgoal: np.ndarray | str, steps_limit=None) -> ValidationResult:
        step: int = 0
        action_path: list[int] = []
        is_solved: bool = False

        if self.env.is_solved(subgoal):
            is_solved = True

        if steps_limit is None:
            steps_limit = self.budget_for_achieving_subgoal

        while step < steps_limit:
            step += 1
            distribution_over_actions: Tensor = self.cllp.get_action(state, subgoal)
            action: int = self.env.distribution_to_action(distribution_over_actions)
            action_path.append(action)
            state: np.ndarray | str = self.env.next_state(state, action)
            if isinstance(state, str):
                if state == subgoal:
                    return ValidationResult(True, is_solved, action_path, step, state)
            elif isinstance(state, np.ndarray):
                if np.array_equal(state, subgoal):
                    return ValidationResult(True, is_solved, action_path, step, state)

        return ValidationResult(False, is_solved, action_path, step, subgoal)


class DummyValidator:
    """
    Dummy validator class.

    Always accepts the subgoal. Used with baseline policy.
    """
    def __init__(self, env: GameEnv) -> None:
        self.env = env

    def construct_network(self) -> None:
        pass

    def is_valid(self, state: np.ndarray, subgoal: np.ndarray, **kwargs) -> ValidationResult:
        return ValidationResult(True, self.env.is_solved(subgoal), [], 1, subgoal)
