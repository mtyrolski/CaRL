import os
from pathlib import Path
from typing import TypeAlias, cast

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
from carl.planners.base import Experience
from carl.solver.nodes import EnvWithRestore
from carl.utils.generator_targets import DEFAULT_K_OFFSETS
from carl.utils.generator_targets import SubgoalTargetMode
from carl.utils.generator_targets import generate_k_offset_generator_targets
from carl.utils.generator_targets import generate_sliding_window_generator_targets

UntokenizedTrajectory: TypeAlias = list[np.ndarray] | list[str]


"""
Data module for preparing game trajectories for various training goals.
Supports policy, value, subgoal, and generator training data.
"""
class GameDataModule(pl.LightningDataModule):
    """
    Prepare and load tokenized game trajectory data for different TrainingGoal types.
    Handles loading raw trajectories, tokenization, train/val split, and DataLoader setup.
    """
    def __init__(
        self,
        env: GameEnv,
        dataset_path: str | Path | None,
        save_tokenized_dataset_path: str,
        training_goal: TrainingGoal | str,
        subgoal_distance_interval: list[int] | None = None,
        experiences_path: str | Path | None = None,
        experiences_glob: str = "*.joblib",
        generator_target_mode: str | None = None,
        generator_k: int | None = None,
        generator_offsets: list[int] | None = None,
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

        self.env = env

        # Determine dataset path if provided
        self.dataset_path: Path | None = None
        if dataset_path is not None:
            self.dataset_path = Path(dataset_path) if not isinstance(dataset_path, Path) else dataset_path

        self.training_goal = (training_goal if isinstance(training_goal, TrainingGoal) else TrainingGoal(training_goal))

        if self.training_goal == TrainingGoal.CLLP:
            assert (subgoal_distance_interval
                    is not None), "Subgoal distance interval must be specified for type of data 'cllp'."
        if self.training_goal == TrainingGoal.GENERATOR:
            assert (untokenized_data or dataset_path or experiences_path), (
                "Please provide dataset_path, untokenized_data, or experiences_path for generator training."
            )
        if self.training_goal != TrainingGoal.GENERATOR:
            assert (untokenized_data or dataset_path), 'Please provide either actual data or a path to its location.'

        self.untokenized_data = untokenized_data
        self.num_of_trajectories = num_of_trajectories
        self.subgoal_distance_interval = subgoal_distance_interval
        self.experiences_path = Path(experiences_path) if experiences_path is not None else None
        self.experiences_glob = experiences_glob
        self.generator_target_mode = generator_target_mode
        self.generator_k = generator_k
        self.generator_offsets = generator_offsets
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
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}

        logger.info(f'Tokenizing data for training goal {self.training_goal.value}.')

        if self.training_goal == TrainingGoal.GENERATOR and self.experiences_path is not None:
            x_tensors, y_tensors = self._generator_tokenize_from_experiences()
        else:
            # Load raw trajectory data into a dictionary
            untokenized_data: dict[int, UntokenizedTrajectory] = self._load_untokenized()

            if self.num_of_trajectories is not None:
                untokenized_data = dict(list(untokenized_data.items())[:self.num_of_trajectories])
            logger.info(f'Number of trajectories: {len(untokenized_data)}')

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
        """Create datasets for specified stage: fit, validate, or test."""
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
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def get_train_dataset(self) -> Dataset:
        assert self._train_dataset is not None
        return self._train_dataset

    def get_val_dataset(self) -> Dataset:
        assert self._val_dataset is not None
        return self._val_dataset

    def get_test_dataset(self) -> Dataset:
        assert self._test_dataset is not None
        return self._test_dataset

    def _load_untokenized(self) -> dict[int, UntokenizedTrajectory]:
        """Load untokenized trajectory data from files or provided dictionary."""
        untokenized_data: dict[int, UntokenizedTrajectory] = {}

        if self.untokenized_data is not None:
            untokenized_data = self.untokenized_data
        elif self.dataset_path is not None and self.dataset_path.is_dir():
            for file in tqdm(self.dataset_path.iterdir()):
                logger.info(f'Loading data from file {file}.')
                part_dict: dict[int, UntokenizedTrajectory] = joblib.load(file)
                untokenized_data.update(part_dict)

                if self.num_of_trajectories is not None and len(untokenized_data) >= self.num_of_trajectories:
                    break
        elif self.dataset_path is not None:
            with open(self.dataset_path, 'rb') as file:
                logger.info(f'Loading data from file {self.dataset_path}.')
                untokenized_data = joblib.load(file)
        else:
            raise ValueError('No dataset path or untokenized data provided.')

        return untokenized_data

    def _flatten_experiences(self, item: object) -> list[Experience]:
        experiences: list[Experience] = []
        stack: list[object] = [item]
        while stack:
            current = stack.pop()
            if isinstance(current, Experience):
                experiences.append(current)
            elif isinstance(current, list):
                stack.extend(current)
        return experiences

    def _load_experiences(self) -> list[Experience]:
        assert self.experiences_path is not None
        if self.experiences_path.is_dir():
            files = sorted(self.experiences_path.glob(self.experiences_glob))
        elif self.experiences_path.is_file():
            files = [self.experiences_path]
        else:
            files = sorted(Path().glob(str(self.experiences_path)))

        experiences: list[Experience] = []
        for file in files:
            loaded = joblib.load(file)
            experiences.extend(self._flatten_experiences(loaded))

        if self.num_of_trajectories is not None:
            experiences = experiences[:self.num_of_trajectories]

        logger.info(f'Loaded experiences: {len(experiences)}')
        return experiences

    def _get_env_with_restore(self) -> EnvWithRestore:
        env = self.env
        missing = [
            name for name in ("restore_full_state_from_np_array_version", "get_state")
            if not hasattr(env, name)
        ]
        if missing:
            raise ValueError(
                "Experience-based generator targets require env to implement "
                f"{', '.join(missing)}."
            )
        return cast(EnvWithRestore, env)

    def _generator_tokenize_from_experiences(
            self, ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        experiences = self._load_experiences()
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}

        mode_value = self.generator_target_mode or SubgoalTargetMode.SLIDING_WINDOW.value
        mode = SubgoalTargetMode(mode_value)
        offsets = list(self.generator_offsets) if self.generator_offsets is not None else list(DEFAULT_K_OFFSETS)

        env = self._get_env_with_restore()
        for idx, exp in enumerate(experiences):
            targets: list[tuple[Tensor, Tensor]]
            if mode == SubgoalTargetMode.SLIDING_WINDOW:
                targets = generate_sliding_window_generator_targets(
                    [exp],
                    env,
                    distance_range=self.subgoal_distance_interval,
                )
            else:
                if self.generator_k is None:
                    raise ValueError('generator_k must be provided for k_offsets mode.')
                generator_k = self.generator_k
                targets = generate_k_offset_generator_targets(
                    [exp],
                    env,
                    k=generator_k,
                    offsets=offsets,
                )

            if not targets:
                continue

            x_tensors[idx] = torch.stack([x for x, _ in targets], dim=0)
            y_tensors[idx] = torch.stack([y for _, y in targets], dim=0)

        return x_tensors, y_tensors

    def _policy_tokenize(
            self, untokenized_data: dict[int, UntokenizedTrajectory]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        x_tensors: dict[int, Tensor] = {}
        y_tensors: dict[int, Tensor] = {}
        for key, trajectory in tqdm(untokenized_data.items()):
            trajectory = trajectory[:self.trajectory_length]
            if len(trajectory) == 0:
                continue
            if isinstance(trajectory[0], np.ndarray):
                trajectory_states = trajectory
            elif isinstance(trajectory[0], str):
                trajectory_states = trajectory
            else:
                continue
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory_states) - 1):
                x: Tensor
                y: Tensor
                action = self.env.detect_action(trajectory_states[position], trajectory_states[position + 1])
                assert action is not None
                x, y = self.env.tokenizer.x_y_tokenizer(trajectory_states[position], action, self.training_goal)
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
            if len(trajectory) == 0:
                continue
            if isinstance(trajectory[0], np.ndarray):
                trajectory_states = trajectory
            elif isinstance(trajectory[0], str):
                trajectory_states = trajectory
            else:
                continue
            trajectory_length: int = len(trajectory_states)
            for position in range(trajectory_length):
                x: Tensor
                y: Tensor
                distance_to_solution: int = trajectory_length - (position + 1)
                x, y = self.env.tokenizer.x_y_tokenizer(
                    trajectory_states[position],
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
        assert self.subgoal_distance_interval is not None
        max_subgoal_distance: int = max(self.subgoal_distance_interval)

        for key, trajectory in tqdm(untokenized_data.items()):
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            trajectory = trajectory[:self.trajectory_length]
            if len(trajectory) == 0:
                continue
            if isinstance(trajectory[0], np.ndarray):
                trajectory_states = trajectory
            elif isinstance(trajectory[0], str):
                trajectory_states = trajectory
            else:
                continue
            trajectory_length: int = len(trajectory_states)
            action_path: list[int] = []
            for i in range(len(trajectory_states) - 1):
                action = self.env.detect_action(trajectory_states[i], trajectory_states[i + 1])
                assert action is not None
                action_path.append(action)

            for p in range(trajectory_length):
                for dist in range(1, max_subgoal_distance + 1):
                    if p + dist >= trajectory_length:
                        break

                    x: Tensor
                    y: Tensor

                    x, y = self.env.tokenizer.x_y_tokenizer(
                        (trajectory_states[p], trajectory_states[p + dist]),
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
            if len(trajectory) == 0:
                continue
            if isinstance(trajectory[0], np.ndarray):
                trajectory_states = trajectory
            elif isinstance(trajectory[0], str):
                trajectory_states = trajectory
            else:
                continue
            trajectory_length: int = len(trajectory_states)

            for position in range(trajectory_length - 1):
                if self.cut_last_subgoals is not None and position == self.cut_last_subgoals:
                    break
                if self.subgoal_distance_interval is None:
                    distance_range = range(1, trajectory_length - position)
                else:
                    distance_range = self.subgoal_distance_interval
                for dist in distance_range:
                    x: Tensor
                    y: Tensor
                    inner_dist: int = min(dist, trajectory_length - 1 - position)

                    x, y = self.env.tokenizer.x_y_tokenizer(
                        trajectory_states[position],
                        trajectory_states[position + inner_dist],
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
            if len(trajectory) == 0:
                continue
            if isinstance(trajectory[0], np.ndarray):
                trajectory_states = trajectory
            elif isinstance(trajectory[0], str):
                trajectory_states = trajectory
            else:
                continue
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory_states) - 1):
                x: Tensor
                y: Tensor
                action = self.env.detect_action(trajectory_states[position], trajectory_states[position + 1])
                assert action is not None
                x, y = self.env.tokenizer.x_y_tokenizer(trajectory_states[position], action, self.training_goal)
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
            if len(trajectory) == 0:
                continue
            if isinstance(trajectory[0], np.ndarray):
                trajectory_states = trajectory
            elif isinstance(trajectory[0], str):
                trajectory_states = trajectory
            else:
                continue
            trajectory_length: int = len(trajectory_states)
            for position in range(trajectory_length):
                x: Tensor
                y: Tensor
                distance_to_solution: int = trajectory_length - (position + 1)
                x, y = self.env.tokenizer.x_y_tokenizer(
                    trajectory_states[position],
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
            if len(trajectory) == 0:
                continue
            if isinstance(trajectory[0], np.ndarray):
                trajectory_states = trajectory
            elif isinstance(trajectory[0], str):
                trajectory_states = trajectory
            else:
                continue
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory_states) - 1):
                x: Tensor
                y: Tensor
                action = self.env.detect_action(trajectory_states[position], trajectory_states[position + 1])
                assert action is not None
                x, y = self.env.tokenizer.x_y_tokenizer(
                    (trajectory_states[position], action),
                    trajectory_states[position + 1],
                    self.training_goal,
                )
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
            if len(trajectory) == 0:
                continue
            if isinstance(trajectory[0], np.ndarray):
                trajectory_states = trajectory
            elif isinstance(trajectory[0], str):
                trajectory_states = trajectory
            else:
                continue
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory_states) - 1):
                x: Tensor
                y: Tensor
                action = self.env.detect_action(trajectory_states[position], trajectory_states[position + 1])
                assert action is not None
                x, y = self.env.tokenizer.x_y_tokenizer(
                    trajectory_states[position],
                    (trajectory_states[position + 1], action),
                    self.training_goal,
                )
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
            if len(trajectory) == 0:
                continue
            if isinstance(trajectory[0], np.ndarray):
                trajectory_states = trajectory
            elif isinstance(trajectory[0], str):
                trajectory_states = trajectory
            else:
                continue
            tem_x_tensor: list[Tensor] = []
            tem_y_tensor: list[Tensor] = []
            for position in range(len(trajectory_states) - 1):
                x: Tensor
                y: Tensor
                action = self.env.detect_action(trajectory_states[position], trajectory_states[position + 1])
                assert action is not None
                x, y = self.env.tokenizer.x_y_tokenizer(
                    trajectory_states[position],
                    (action, trajectory_states[position + 1]),
                    self.training_goal,
                )
                tem_x_tensor.append(x)
                tem_y_tensor.append(y)

            x_tensors[key] = torch.cat(tem_x_tensor, dim=0)
            y_tensors[key] = torch.cat(tem_y_tensor, dim=0)

        return x_tensors, y_tensors
