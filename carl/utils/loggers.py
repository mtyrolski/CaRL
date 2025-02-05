import copy
import os
from abc import ABC
from abc import abstractmethod
from typing import Any

from loguru import logger

import neptune
import numpy as np
import torch
from omegaconf import ListConfig
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers import TrainerCallback
from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel, TrainerCallback
from transformers.integrations import NeptuneCallback

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal


class CaRLLogger(ABC):
    """Abstract class for custom logger."""
    @abstractmethod
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
        self.tags = tags if isinstance(tags, str) else list(tags)
        if api_token is None:
            self.api_token = os.getenv('NEPTUNE_API_TOKEN', neptune.ANONYMOUS_API_TOKEN)
            logger.success('Retrieved NEPTUNE_API_TOKEN from env.')
        else:
            self.api_token = api_token
        print(f'Using Neptune API token: {self.api_token}')
        self.log_parameters = log_parameters

        if not isinstance(self.tags, list):
            self.tags = OmegaConf.to_container(self.tags)

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
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: PreTrainedModel | None = None

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        self.model = copy.deepcopy(kwargs['model'])
        self.model.to(self.device)
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
            state: np.ndarray | str = self.env.next_state(state, action)

            if isinstance(state, str):
                if state == subgoal:
                    return True, step
            elif isinstance(state, np.ndarray):
                if np.array_equal(state, subgoal):
                    return True, step

        return False, step

    def get_action(
        self,
        state: np.ndarray | str,
        state_after_k: np.ndarray | str,
    ) -> Tensor:

        encoded_boards: Tensor
        encoded_boards, _ = self.env.tokenizer.x_y_tokenizer(
            x=(state, state_after_k),
            y=0,
            training_goal=TrainingGoal.CLLP,
        )

        encoded_boards = encoded_boards.to(self.device)
        with torch.no_grad():
            output: torch.Tensor = self.model(encoded_boards).logits
        return output.softmax(dim=-1)[0]
