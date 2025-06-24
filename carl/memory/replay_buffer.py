from abc import ABC
from typing import Union

import numpy as np
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import tqdm

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.solver.nodes import SearchTreeNode
from carl.solver.nodes import get_solving_path_data


class UniversalReplayBuffer(ABC):
    """
    Interface of general replay buffer for experience replay.

    Handles data replay for each component.
    """
    def add(self, data: SearchTreeNode) -> None:
        raise NotImplementedError()

    def sample_for_generator(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError()

    def sample_for_value(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError()

    def sample_for_policy(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError()


class SimpleUniversalReplayBuffer(UniversalReplayBuffer):
    """
    Simple implementation of the universal replay buffer.

    Stores a separate instance of the buffer for each component.
    """
    def __init__(
        self,
        generator_buffer: Union['SolvingPathGeneratorReplayBuffer', dict[int, 'SolvingPathGeneratorReplayBuffer']],
        value_buffer: 'SolvingPathValueReplayBuffer',
        cllp_buffer: 'SolvingPathConditionalLowLevelPolicyReplayBuffer',
    ) -> None:
        super().__init__()
        self.generator_buffer = generator_buffer
        self.value_buffer = value_buffer
        self.cllp_buffer = cllp_buffer

        if not isinstance(self.generator_buffer, SolvingPathGeneratorReplayBuffer):
            self.generator_buffer = OmegaConf.to_container(self.generator_buffer)

    def add_metrics(self, buffer, buffer_name, metrics, data):
        size_before = len(buffer)
        buffer.add(data)
        return {f'{buffer_name}/size': len(buffer), f'{buffer_name}/size_changed': len(buffer) - size_before}

    def add(self, data: tuple[dict, dict]) -> dict[str, int]:
        metrics = {}
        if isinstance(self.generator_buffer, SolvingPathGeneratorReplayBuffer):
            metrics.update(self.add_metrics(self.generator_buffer, 'generator', metrics, data))
        else:
            assert isinstance(self.generator_buffer, dict)
            for k, buffer in self.generator_buffer.items():
                metrics.update(self.add_metrics(buffer, f'generator_{k}', metrics, data))

        metrics.update(self.add_metrics(self.value_buffer, 'value', metrics, data))
        metrics.update(self.add_metrics(self.cllp_buffer, 'cllp', metrics, data))
        return metrics

    def add_from_trajectories(self, trajectories: list[np.ndarray]):
        """
        Add trajectories to the replay buffer.

        :param trajectories: list of trajectories presented as ndarray of shape list[(trajectory_length, *state_dim)]
        :return: None
        """

        for trajectory in tqdm(trajectories):
            self.value_buffer.add_from_trajectories([trajectory])
            self.cllp_buffer.add_from_trajectories([trajectory])
            if isinstance(self.generator_buffer, SolvingPathGeneratorReplayBuffer):
                self.generator_buffer.add_from_trajectories([trajectory])
            else:
                assert isinstance(self.generator_buffer, dict)
                for buffer in self.generator_buffer.values():
                    buffer.add_from_trajectories([trajectory])

    def sample_for_generator(self, batch_size, k: int | None = None):
        if k is None:
            return self.generator_buffer.sample(batch_size)
        else:
            return self.generator_buffer[k].sample(batch_size)

    def sample_for_value(self, batch_size):
        return self.value_buffer.sample(batch_size)

    def sample_for_policy(self, batch_size):
        return self.cllp_buffer.sample(batch_size)

    def get_buffer_for_generator(self, k: int | None):
        if k is None:
            return self.generator_buffer
        else:
            return self.generator_buffer[k]

    def get_buffer_for_value(self):
        return self.value_buffer

    def get_buffer_for_policy(self):
        return self.cllp_buffer


class ReplayBuffer:
    """
    Interface of replay buffer for experience replay.

    Handles data replay for a single component.
    """
    def add(self, data: SearchTreeNode) -> None:
        raise NotImplementedError()

    def sample(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError()


class OfflineReplayBuffer(ReplayBuffer):
    """
    Replay buffer for experience replay.

    Stores fixed datapoints for training.
    """
    def __init__(self, max_size: int) -> None:
        super().__init__()
        self.buffer = []
        self.max_size = max_size

    def add(self, data: SearchTreeNode) -> None:
        raise NotImplementedError()

    def sample(self, batch_size: int) -> np.ndarray:
        return np.random.choice(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class SolvingPathGeneratorReplayBuffer(OfflineReplayBuffer):
    """
    Offline replay buffer for subgoal generator.

    Stores training datapoints extracted from the solving path.
    """
    def __init__(self, max_size: int, distance_range: list[int], env: GameEnv):
        super().__init__(max_size)

        self.distance_range = distance_range
        self.env = env
        self.training_goal: TrainingGoal = TrainingGoal.GENERATOR

    def add(self, data: tuple[dict, dict]):
        _, search_info = data
        if search_info['solving_node'] is None:
            return
        _, _, _, _, state_path = get_solving_path_data(search_info['solving_node'], include_state_path=True, env=self.env)

        for xy in self.preprocess_trajectory(state_path):
            self.buffer.append(xy)

            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)

    def add_from_trajectories(self, trajectories: list[np.ndarray]):
        """
        Add trajectories to the replay buffer.

        :param trajectories: list of trajectories presented as ndarray of shape list[(trajectory_length, *state_dim)]
        :return: None
        """
        for trajectory in trajectories:
            for xy in self.preprocess_trajectory(trajectory):
                self.buffer.append(xy)

    def preprocess_trajectory(self, trajectory: list[np.ndarray]) -> list[tuple[Tensor, Tensor]]:
        preprocess_trajectory: list[tuple[Tensor, Tensor]] = []
        trajectory_length: int = len(trajectory)

        for i in range(trajectory_length - 1):
            for dist in self.distance_range:
                inner_dist: int = min(dist, trajectory_length - 1 - i)
                x: Tensor
                y: Tensor

                x, y = self.env.tokenizer.x_y_tokenizer(
                    x=trajectory[i],
                    y=trajectory[i + inner_dist],
                    training_goal=self.training_goal,
                )

                preprocess_trajectory.append((x, y))

                if i + inner_dist >= len(trajectory) - 1:
                    # don't add more than one copy of the last subgoal
                    break

        return preprocess_trajectory


class SolvingPathValueReplayBuffer(OfflineReplayBuffer):
    """
    Offline replay buffer for value function.

    Implements a simple supervised objective for learning value function.
    Stores training datapoints extracted from the solving path.
    """
    def __init__(self, max_size: int, env: GameEnv):
        super().__init__(max_size)
        self.env = env
        self.training_goal: TrainingGoal = TrainingGoal.VALUE

    def add(self, data: tuple[dict, dict]) -> None:
        search_info: dict
        _, search_info = data
        if search_info['solving_node'] is None:
            return

        state_path: list[np.ndarray]
        _, _, _, _, state_path = get_solving_path_data(search_info['solving_node'], include_state_path=True, env=self.env)

        for x, y in self.preprocess_trajectory(state_path):
            self.buffer.append((x, y))

            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)

    def add_from_trajectories(self, trajectories: list[np.ndarray]):
        """
        Add trajectories to the replay buffer.

        :param trajectories: list of trajectories presented as ndarray of shape list[(trajectory_length, *state_dim)]
        :return: None
        """
        for trajectory in trajectories:
            for xy in self.preprocess_trajectory(trajectory):
                self.buffer.append(xy)

                if len(self.buffer) > self.max_size:
                    self.buffer.pop(0)

    def preprocess_trajectory(self, trajectory: list[np.ndarray]) -> list[tuple[Tensor, Tensor]]:
        preprocess_trajectory: list[tuple[Tensor, Tensor]] = []
        trajectory_length: int = len(trajectory)

        for position in range(trajectory_length):
            distance_to_solution: int = trajectory_length - (position + 1)
            x: Tensor
            y: Tensor

            x, y = self.env.tokenizer.x_y_tokenizer(x=trajectory[position],
                                                    y=distance_to_solution,
                                                    training_goal=self.training_goal)

            preprocess_trajectory.append((x, y))

        return preprocess_trajectory


class SolvingPathConditionalLowLevelPolicyReplayBuffer(OfflineReplayBuffer):
    """
    Offline replay buffer for policy.

    Implements simple behavioral cloning.
    Stores training datapoints extracted from the solving path.
    """
    def __init__(self, max_size: int, max_distance: int, env) -> None:
        super().__init__(max_size)
        self.max_distance = max_distance
        self.env = env
        self.training_goal: TrainingGoal = TrainingGoal.CLLP

    def add(self, data: tuple[dict, dict]) -> None:
        search_info: dict
        _, search_info = data
        if search_info['solving_node'] is None:
            return

        action_path: list[int]
        state_path: list[np.ndarray]
        _, action_path, _, _, state_path = get_solving_path_data(search_info['solving_node'],
                                                              include_state_path=True,
                                                              env=self.env)
        for xy in self.preprocess_trajectory(state_path, action_path):
            self.buffer.append(xy)

            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)

    def add_from_trajectories(self, trajectories: list[np.ndarray]):
        """
        Add trajectories to the replay buffer.

        :param trajectories: list of trajectories presented as ndarray of shape list[(trajectory_length, *state_dim)]
        :return: None
        """
        for trajectory in trajectories:
            action_path: list[int] = []

            for i in range(len(trajectory) - 1):
                action_path.append(self.env.detect_action(trajectory[i], trajectory[i + 1]))

            for xy in self.preprocess_trajectory(trajectory, action_path):
                self.buffer.append(xy)

                if len(self.buffer) > self.max_size:
                    self.buffer.pop(0)

    def preprocess_trajectory(self, state_path: list[np.ndarray],
                              action_path: list[int]) -> list[tuple[Tensor, Tensor]]:
        preprocess_trajectory: list[tuple[Tensor, Tensor]] = []

        for i in range(len(state_path)):
            for dist in range(1, self.max_distance + 1):
                if i + dist >= len(state_path):
                    break

                x: Tensor
                y: Tensor
                x, y = self.env.tokenizer.x_y_tokenizer(
                    x=(state_path[i], state_path[i + dist]),
                    y=action_path[i],
                    training_goal=self.training_goal,
                )
                preprocess_trajectory.append((x, y))

        return preprocess_trajectory


class GlobalPolicyReplayBuffer(OfflineReplayBuffer):
    """
    Offline replay buffer for policy.

    Implements simple behavioral cloning.
    Stores training datapoints extracted from the whole search tree.
    """
    def __init__(self, max_size: int, max_distance: int, env: GameEnv) -> None:
        super().__init__(max_size)
        self.max_distance = max_distance
        self.env = env
