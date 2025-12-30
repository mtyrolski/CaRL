import copy
import os
from abc import ABC
from abc import abstractmethod
from typing import Any, cast

import neptune
import numpy as np
import torch
from loguru import logger
from omegaconf import ListConfig
from torch import Tensor
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from transformers.integrations.integration_utils import NeptuneCallback

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.utils.torch_device import resolve_device


class CaRLLogger(ABC):
    """Abstract class for custom logger."""
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def return_logger(self) -> Any:
        raise NotImplementedError


class NeptuneCaRLLogger(CaRLLogger):
    """Custom logger for Neptune."""
    def __init__(
        self,
        name: str,
        description: str,
        project: str,
        tags: str | ListConfig,
        log_parameters: bool,
        api_token: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.description = description
        self.project = project
        self.tags: str | list[str]
        if isinstance(tags, str):
            self.tags = tags
        else:
            self.tags = [str(tag) for tag in list(tags)]
        if api_token is None:
            self.api_token = os.getenv('NEPTUNE_API_TOKEN', neptune.ANONYMOUS_API_TOKEN)
            logger.success('Retrieved NEPTUNE_API_TOKEN from env.')
        else:
            self.api_token = api_token
        print(f'Using Neptune API token: {self.api_token}')
        self.log_parameters = log_parameters

        self.run = neptune.init_run(
            name=self.name,
            description=self.description,
            tags=self.tags,
            project=self.project,
            api_token=self.api_token,
        )

    def return_logger(self) -> NeptuneCallback:
        """Returns the Neptune logger."""

        return NeptuneCallback(run=self.run, log_parameters=self.log_parameters)


class CLLPTestLogger(TrainerCallback):
    def __init__(
        self,
        inner_logger: Any,
        data_to_evaluate: dict[int, list[np.ndarray]],
        distance_range: list[int],
        env: GameEnv,
    ) -> None:
        super().__init__()
        self.data_to_evaluate = data_to_evaluate
        self.distance_range: list[int] = distance_range
        self.inner_logger = inner_logger
        self.budget_for_achieving_subgoal: int = max(distance_range) + 2
        self.env = env
        self.step: int = 0
        self.device: torch.device = resolve_device()
        self.model: PreTrainedModel | None = None

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        self.model = cast(PreTrainedModel, copy.deepcopy(kwargs['model']))
        assert self.model is not None
        model_module = cast(torch.nn.Module, self.model)
        model_module.to(self.device)
        self.model = cast(PreTrainedModel, model_module)
        self.model.eval()

        cllp_achieved_goals: float
        cllp_distances_to_goal: float

        cllp_achieved_goals, cllp_distances_to_goal = self.test_cllp()

        self.inner_logger.run['mean_cllp_achieved_goals'].log(step=self.step, value=cllp_achieved_goals)
        self.inner_logger.run['mean_cllp_distances_to_goal'].log(step=self.step, value=cllp_distances_to_goal)

        self.step += 1

    def test_cllp_on_trajectory(self, trajectory: list[np.ndarray]) -> tuple[float, float]:

        trajectory_length: int = len(trajectory)
        distance_range: list[int] = self.distance_range
        cllp_achieved_goals: int = 0
        distances_to_goal: int = 0
        num_of_pairs: int = 0

        for i in range(trajectory_length - 1):
            for dist in distance_range:
                inner_dist: int = min(dist, trajectory_length - 1 - i)
                x: np.ndarray
                y: np.ndarray

                x = trajectory[i]
                y = trajectory[i + inner_dist]

                achieved: int
                distance: int
                achieved, distance = self.is_valid(x, y)
                num_of_pairs += 1
                cllp_achieved_goals += int(achieved)
                distances_to_goal += distance

                if i + inner_dist >= len(trajectory) - 1:
                    # don't add more than one copy of the last subgoal
                    break

        mean_cllp_achieved_goals: float = cllp_achieved_goals / num_of_pairs
        mean_distances_to_goal: float = distances_to_goal / num_of_pairs

        return mean_cllp_achieved_goals, mean_distances_to_goal

    def test_cllp(self) -> tuple[float, float]:
        cllp_achieved_goals: float = 0.0
        cllp_distances_to_goal: float = 0.0
        mean_cllp_achieved_goals: float
        mean_distances_to_goal: float

        for trajectory in tqdm(self.data_to_evaluate.values()):
            mean_cllp_achieved_goals, mean_distances_to_goal = self.test_cllp_on_trajectory(trajectory)
            cllp_achieved_goals += mean_cllp_achieved_goals
            cllp_distances_to_goal += mean_distances_to_goal

        return cllp_achieved_goals / len(self.data_to_evaluate), cllp_distances_to_goal / len(self.data_to_evaluate)

    def is_valid(self, state, subgoal) -> tuple[bool, int]:
        step: int = 0

        while step < self.budget_for_achieving_subgoal:
            step += 1
            distribution_over_actions: Tensor = self.get_action(state, subgoal)
            action: int = self.env.distribution_to_action(distribution_over_actions)
            next_state: np.ndarray | str = self.env.next_state(state, action)

            if isinstance(next_state, str):
                if next_state == subgoal:
                    return True, step
            elif isinstance(next_state, np.ndarray):
                if np.array_equal(next_state, subgoal):
                    return True, step

        return False, step

    def get_action(
        self,
        state: np.ndarray | str,
        state_after_k: np.ndarray | str,
    ) -> Tensor:

        encoded_boards: Tensor
        x_value: tuple[np.ndarray, np.ndarray] | tuple[str, str]
        if isinstance(state, np.ndarray) and isinstance(state_after_k, np.ndarray):
            x_value = (state, state_after_k)
        elif isinstance(state, str) and isinstance(state_after_k, str):
            x_value = (state, state_after_k)
        else:
            raise ValueError("State and subgoal types must match for CLLP evaluation.")

        encoded_boards, _ = self.env.tokenizer.x_y_tokenizer(
            x=x_value,
            y=0,
            training_goal=TrainingGoal.CLLP,
        )

        encoded_boards = encoded_boards.to(self.device)
        with torch.no_grad():
            assert self.model is not None
            output: torch.Tensor = self.model(encoded_boards).logits
        return output.softmax(dim=-1)[0]

def log_error_and_raise(message: str, exception_cls: type[Exception] = ValueError) -> None:
    """Logs an error message and raises a RuntimeError."""
    logger.error(message)
    raise exception_cls(message)
