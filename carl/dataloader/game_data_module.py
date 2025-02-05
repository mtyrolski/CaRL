import os
from pathlib import Path
from typing import TypeAlias

import joblib
import lightning as pl
import numpy as np
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from carl.dataloader.game_dataset import GameDataset
from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal

UntokenizedTrajectory: TypeAlias = list[np.ndarray] | list[str]


class GameDataModule(pl.LightningDataModule):
    """Data module for training a TrainingGoal (which can be: value network, policy network, conditional low level policy network, or generator)."""
    def __init__(
        self,
        env: GameEnv,
        dataset_path: str | Path | None,
        save_tokenized_dataset_path: str,
        training_goal: TrainingGoal | str,
        subgoal_distance_interval: list[int] | None = None,
        untokenized_data: dict[int, UntokenizedTrajectory] | None = None,
        num_of_trajectories: int | None = None,
        validation_split: float = 0.1,
        batch_size: int = 1,
        trajectory_length: int | None = None,
        num_workers: int = 1,
        for_testing: bool = False,
        cut_last_subgoals: int | None = None,
    ) -> None:
        super().__init__()

        if training_goal in [TrainingGoal.CLLP, TrainingGoal.GENERATOR]:
            assert (subgoal_distance_interval
                    is not None), "Subgoal distance interval must be specified for type of data 'cllp' or 'generator'."

        self.env = env
        assert (untokenized_data or dataset_path), 'Please provide either actual data or a path to its location.'

        if dataset_path is None:
            self.dataset_path = None
        else:
            self.dataset_path = (dataset_path if isinstance(dataset_path, Path) else Path(dataset_path))

        self.training_goal = (training_goal if isinstance(training_goal, TrainingGoal) else TrainingGoal(training_goal))

        self.untokenized_data = untokenized_data
        self.num_of_trajectories = num_of_trajectories
        self.subgoal_distance_interval = subgoal_distance_interval
        self.validation_split = validation_split
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.save_tokenized_dataset_path = Path(save_tokenized_dataset_path + f'_{self.training_goal.value}' +
                                                f'_{str(np.random.randint(10000000))}')

        os.makedirs(self.save_tokenized_dataset_path, exist_ok=True)

        self._for_testing = for_testing
        self.cut_last_subgoals = cut_last_subgoals

        self._train_dataset: GameDataset | None = None
        self._val_dataset: GameDataset | None = None
        self._test_dataset: GameDataset | None = None

    def prepare_data(self) -> None:
        untokenized_data: dict[int, UntokenizedTrajectory] = {}

        if self.untokenized_data is not None:
            untokenized_data = self.untokenized_data

        elif self.dataset_path.is_dir():
            for file in tqdm(self.dataset_path.iterdir()):
                logger.info(f'Loading data from file {file}.')
                part_dict: dict[int, UntokenizedTrajectory] = joblib.load(file)
                untokenized_data.update(part_dict)

                if self.num_of_trajectories is not None and len(untokenized_data) >= self.num_of_trajectories:
                    break
        else:
            with open(self.dataset_path, 'rb') as file:
                logger.info(f'Loading data from file {self.dataset_path}.')
                untokenized_data = joblib.load(file)

        untokenized_data = dict(list(untokenized_data.items())[:self.num_of_trajectories])
        logger.info(f'Number of trajectories: {len(untokenized_data)}')

        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}

        logger.info(f'Tokenizing data for training goal {self.training_goal.value}.')

        match self.training_goal:
            case TrainingGoal.POLICY:
                x_tensors, y_tensors = self._policy_tokenize(untokenized_data)
            case TrainingGoal.VALUE:
                x_tensors, y_tensors = self._value_tokenize(untokenized_data)
            case TrainingGoal.CLLP:
                x_tensors, y_tensors = self._cllp_tokenize(untokenized_data)
            case TrainingGoal.GENERATOR:
                x_tensors, y_tensors = self._generator_tokenize(untokenized_data)
            case TrainingGoal.POLICY_GENERATION:
                x_tensors, y_tensors = self._policy_generation_tokenize(untokenized_data)
            case TrainingGoal.VALUE_GENERATION:
                x_tensors, y_tensors = self._value_generation_tokenize(untokenized_data)
            case TrainingGoal.STATE_ACTION_STATE:
                x_tensors, y_tensors = self._state_action_state(untokenized_data)
            case TrainingGoal.STATE_STATE_ACTION:
                x_tensors, y_tensors = self._state_state_action(untokenized_data)
            case TrainingGoal.STATE_ACTION_STATE_GENERATOR:
                x_tensors, y_tensors = self._state_action_state_generator_(untokenized_data)

        assert len(x_tensors) == len(y_tensors), 'x and y tensors must be of same length.'
        assert len(x_tensors) != 0, ('No data was tokenized. If you are preparing data for a generator or a '
                                     'conditional low level policy, please make sure that dataset contains '
                                     'trajectories which have lengths greater than max subgoal distance.')

        if self._for_testing:

            keys = list(x_tensors.keys())

            testing_x = torch.stack([x_step for key in keys for x_step in x_tensors[key]], dim=0)
            testing_y = torch.stack([y_step for key in keys for y_step in y_tensors[key]], dim=0)

            joblib.dump(
                [testing_x, testing_y],
                os.path.join(
                    self.save_tokenized_dataset_path,
                    f'{self.env.name}_{self.training_goal.value}_tokenized_all_x_y',
                ),
            )
            return

        train_keys: list[int]
        val_keys: list[int]

        train_keys, val_keys = train_test_split(list(x_tensors.keys()), test_size=self.validation_split)

        training_x = torch.stack([x_step for key in train_keys for x_step in x_tensors[key]], dim=0)
        training_y = torch.stack([y_step for key in train_keys for y_step in y_tensors[key]], dim=0)
        val_x = torch.stack([x_step for key in val_keys for x_step in x_tensors[key]], dim=0)
        val_y = torch.stack([y_step for key in val_keys for y_step in y_tensors[key]], dim=0)

        logger.info(f'Saving tokenized data for training goal {self.training_goal.value}.')
        logger.info(f'Size of training set: {len(training_x)}')
        logger.info(f'Size of validation set: {len(val_x)}')

        joblib.dump(
            [training_x, training_y],
            os.path.join(
                self.save_tokenized_dataset_path,
                f'{self.env.name}_{self.training_goal.value}_tokenized_train_x_y',
            ),
        )
        joblib.dump(
            [val_x, val_y],
            os.path.join(
                self.save_tokenized_dataset_path,
                f'{self.env.name}_{self.training_goal.value}_tokenized_val_x_y',
            ),
        )

        logger.info(f"Creating folder for saving net's weights."
                    f' Please note the name of the folder corresponding to this training'
                    f'goal. In this case: {self.training_goal.value}.')
        os.makedirs(os.path.join('.', 'models_weights', self.training_goal.value), exist_ok=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == 'fit':
            x_train, y_train = joblib.load(
                os.path.join(
                    self.save_tokenized_dataset_path,
                    f'{self.env.name}_{self.training_goal.value}_tokenized_train_x_y',
                ))
            self._train_dataset = GameDataset(x_train, y_train)

        if stage in ['fit', 'validate']:
            x_val, y_val = joblib.load(
                os.path.join(
                    self.save_tokenized_dataset_path,
                    f'{self.env.name}_{self.training_goal.value}_tokenized_val_x_y',
                ))
            self._val_dataset = GameDataset(x_val, y_val)

        if stage in ['predict']:
            raise NotImplementedError

        if stage in ['test']:
            x_all, y_all = joblib.load(
                os.path.join(
                    self.save_tokenized_dataset_path,
                    f'{self.env.name}_{self.training_goal.value}_tokenized_all_x_y',
                ))

            self._test_dataset = GameDataset(x_all, y_all)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def get_train_dataset(self) -> Dataset:
        return self._train_dataset

    def get_val_dataset(self) -> Dataset:
        return self._val_dataset

    def get_test_dataset(self) -> Dataset:
        raise self._test_dataset

    def _policy_tokenize(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}
        for key, trajectory in tqdm(untokenized_data.items()):
            trajectory = trajectory[:self.trajectory_length]
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory) - 1):
                x: Tensor
                y: Tensor
                action: int = self.env.detect_action(trajectory[position], trajectory[position + 1])
                x, y = self.env.tokenizer.x_y_tokenizer(trajectory[position], action, self.training_goal)
                tem_x_tensor.append(x)
                tem_y_tensor.append(y)
            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)
        return x_tensors, y_tensors

    def _value_tokenize(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}

        for key, trajectory in tqdm(untokenized_data.items()):
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            trajectory = trajectory[:self.trajectory_length]
            trajectory_length: int = len(trajectory)
            for position in range(trajectory_length):
                x: Tensor
                y: Tensor
                distance_to_solution: int = trajectory_length - (position + 1)
                x, y = self.env.tokenizer.x_y_tokenizer(
                    trajectory[position],
                    distance_to_solution,
                    self.training_goal,
                )
                tem_x_tensor.append(x)
                tem_y_tensor.append(y)
            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)

        return x_tensors, y_tensors

    def _cllp_tokenize(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}
        max_subgoal_distance: int = max(self.subgoal_distance_interval)

        for key, trajectory in tqdm(untokenized_data.items()):
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            trajectory = trajectory[:self.trajectory_length]
            trajectory_length: int = len(trajectory)
            action_path: list[int] = [
                self.env.detect_action(trajectory[i], trajectory[i + 1]) for i in range(len(trajectory) - 1)
            ]

            for p in range(trajectory_length):
                for dist in range(1, max_subgoal_distance + 1):
                    if p + dist >= trajectory_length:
                        break

                    x: Tensor
                    y: Tensor

                    x, y = self.env.tokenizer.x_y_tokenizer(
                        (
                            trajectory[p],
                            trajectory[p + dist],
                        ),
                        action_path[p],
                        self.training_goal,
                    )

                    tem_x_tensor.append(x)
                    tem_y_tensor.append(y)

            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)

        return x_tensors, y_tensors

    def _generator_tokenize(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}

        for key, trajectory in tqdm(untokenized_data.items()):
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            trajectory = trajectory[:self.trajectory_length]
            trajectory_length: int = len(trajectory)

            for position in range(trajectory_length - 1):
                if self.cut_last_subgoals is not None and position == self.cut_last_subgoals:
                    break
                for dist in self.subgoal_distance_interval:
                    x: Tensor
                    y: Tensor
                    inner_dist: int = min(dist, trajectory_length - 1 - position)

                    x, y = self.env.tokenizer.x_y_tokenizer(
                        trajectory[position],
                        trajectory[position + inner_dist],
                        self.training_goal,
                    )
                    tem_x_tensor.append(x)
                    tem_y_tensor.append(y)

                    if position + inner_dist >= trajectory_length - 1:
                        # don't add more than one copy of the last subgoal
                        break

            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)

        return x_tensors, y_tensors

    def _policy_generation_tokenize(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}
        for key, trajectory in tqdm(untokenized_data.items()):
            trajectory = trajectory[:self.trajectory_length]
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory) - 1):
                x: Tensor
                y: Tensor
                action: int = self.env.detect_action(trajectory[position], trajectory[position + 1])
                x, y = self.env.tokenizer.x_y_tokenizer(trajectory[position], action, self.training_goal)
                tem_x_tensor.append(x)
                tem_y_tensor.append(y)
            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)
        return x_tensors, y_tensors

    def _value_generation_tokenize(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}

        for key, trajectory in tqdm(untokenized_data.items()):
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            trajectory = trajectory[:self.trajectory_length]
            trajectory_length: int = len(trajectory)
            for position in range(trajectory_length):
                x: Tensor
                y: Tensor
                distance_to_solution: int = trajectory_length - (position + 1)
                x, y = self.env.tokenizer.x_y_tokenizer(
                    trajectory[position],
                    distance_to_solution,
                    self.training_goal,
                )
                tem_x_tensor.append(x)
                tem_y_tensor.append(y)
            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)

        return x_tensors, y_tensors

    def _state_action_state(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}

        for key, trajectory in tqdm(untokenized_data.items()):
            trajectory = trajectory[:self.trajectory_length]
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory) - 1):
                x: Tensor
                y: Tensor
                action: int = self.env.detect_action(trajectory[position], trajectory[position + 1])
                x, y = self.env.tokenizer.x_y_tokenizer((trajectory[position], action), trajectory[position + 1],
                                                        self.training_goal)
                tem_x_tensor.append(x)
                tem_y_tensor.append(y)

            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)

        return x_tensors, y_tensors

    def _state_state_action(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}

        for key, trajectory in tqdm(untokenized_data.items()):
            trajectory = trajectory[:self.trajectory_length]
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory) - 1):
                x: Tensor
                y: Tensor
                action: int = self.env.detect_action(trajectory[position], trajectory[position + 1])
                x, y = self.env.tokenizer.x_y_tokenizer(trajectory[position], (trajectory[position + 1], action),
                                                        self.training_goal)
                tem_x_tensor.append(x)
                tem_y_tensor.append(y)

            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)

        return x_tensors, y_tensors

    def _state_action_state_generator_(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}

        for key, trajectory in tqdm(untokenized_data.items()):
            trajectory = trajectory[:self.trajectory_length]
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory) - 1):
                x: Tensor
                y: Tensor
                action: int = self.env.detect_action(trajectory[position], trajectory[position + 1])
                x, y = self.env.tokenizer.x_y_tokenizer(trajectory[position], (action, trajectory[position + 1]),
                                                        self.training_goal)
                tem_x_tensor.append(x)
                tem_y_tensor.append(y)

            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)

        return x_tensors, y_tensors
