from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeAlias

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
from carl.dataloader.universal_generator_types import UniversalGeneratorBatch
from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal

UntokenizedTrajectory: TypeAlias = list[np.ndarray] | list[str] | np.ndarray


class UniversalGeneratorDataModule(pl.LightningDataModule):
    """Recipe-aware datamodule for universal propositional generator training.

    Recipes:
    - `raw`: trajectory-derived proposition targets (no teacher labels, no explicit k labels in outputs)
    - `finetune`: teacher proposition annotations
    - `contrastive`: same teacher proposition annotations, consumed with additional contrastive loss
    """

    def __init__(
        self,
        env: GameEnv,
        recipe: str,
        save_tokenized_dataset_path: str,
        *,
        raw_dataset_path: str | Path | None = None,
        teacher_dataset_path: str | Path | None = None,
        teacher_dataset_glob: str = "*.joblib",
        num_of_trajectories: int | None = None,
        max_samples: int | None = None,
        subgoal_distance_interval: list[int] | None = None,
        validation_split: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 0,
        trajectory_length: int | None = None,
        pad_token_id: int = 1,
        for_testing: bool = False,
        teacher_accept_only: bool = True,
        teacher_reached_only: bool = False,
    ) -> None:
        super().__init__()
        if recipe not in {"raw", "finetune", "contrastive"}:
            raise ValueError(f"Unsupported recipe: {recipe}")

        self.env = env
        self.recipe = recipe
        self.training_goal = TrainingGoal.GENERATOR
        self.raw_dataset_path = Path(raw_dataset_path) if raw_dataset_path is not None else None
        self.teacher_dataset_path = Path(teacher_dataset_path) if teacher_dataset_path is not None else None
        self.teacher_dataset_glob = teacher_dataset_glob
        self.num_of_trajectories = num_of_trajectories
        self.max_samples = max_samples
        self.subgoal_distance_interval = subgoal_distance_interval
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trajectory_length = trajectory_length
        self.pad_token_id = pad_token_id
        self.for_testing = for_testing
        self.teacher_accept_only = teacher_accept_only
        self.teacher_reached_only = teacher_reached_only

        suffix = int(np.random.randint(10_000_000))
        self.save_tokenized_dataset_path = Path(f"{save_tokenized_dataset_path}_{recipe}_{suffix}")
        os.makedirs(self.save_tokenized_dataset_path, exist_ok=True)

        self._train_dataset: GameDataset | None = None
        self._val_dataset: GameDataset | None = None
        self._test_dataset: GameDataset | None = None

    def _iter_raw_trajectories(self) -> list[tuple[int, UntokenizedTrajectory]]:
        if self.raw_dataset_path is None:
            raise ValueError("raw_dataset_path is required for raw recipe")

        data: dict[int, UntokenizedTrajectory] = {}
        if self.raw_dataset_path.is_dir():
            for file in sorted(self.raw_dataset_path.iterdir()):
                logger.info(f"Loading raw trajectories from {file}")
                part = joblib.load(file)
                data.update(part)
                if self.num_of_trajectories is not None and len(data) >= self.num_of_trajectories:
                    break
        else:
            data = joblib.load(self.raw_dataset_path)

        items = list(data.items())
        if self.num_of_trajectories is not None:
            items = items[:self.num_of_trajectories]
        return items

    def _iter_teacher_annotations(self) -> list[dict[str, Any]]:
        if self.teacher_dataset_path is None:
            raise ValueError("teacher_dataset_path is required for finetune/contrastive recipes")

        payloads: list[dict[str, Any]] = []
        if self.teacher_dataset_path.is_dir():
            files = sorted(self.teacher_dataset_path.glob(self.teacher_dataset_glob))
        else:
            files = [self.teacher_dataset_path]

        for file in files:
            logger.info(f"Loading teacher annotations from {file}")
            obj = joblib.load(file)
            if isinstance(obj, dict) and "annotations" in obj:
                anns = obj["annotations"]
            elif isinstance(obj, list):
                anns = obj
            else:
                continue
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                if self.teacher_accept_only and not bool(ann.get("validator_accept", False)):
                    continue
                if self.teacher_reached_only and not bool(ann.get("reached", False)):
                    continue
                payloads.append(ann)
                if self.max_samples is not None and len(payloads) >= self.max_samples:
                    return payloads
        return payloads

    def _tokenize_pair(self, state: Any, proposition: Any) -> tuple[Tensor, Tensor]:
        x, y = self.env.tokenizer.x_y_tokenizer(state, proposition, self.training_goal)
        # x and y are typically shaped (1, seq_len)
        return x, y

    def _tokenize_raw(self) -> tuple[Tensor, Tensor]:
        x_steps: list[Tensor] = []
        y_steps: list[Tensor] = []
        count = 0

        items = self._iter_raw_trajectories()
        for _, trajectory in tqdm(items):
            traj = trajectory[:self.trajectory_length] if self.trajectory_length is not None else trajectory
            if len(traj) == 0:
                continue
            if isinstance(traj, np.ndarray):
                trajectory_states = traj
            elif isinstance(traj[0], (np.ndarray, str)):  # type: ignore[index]
                trajectory_states = traj
            else:
                continue

            trajectory_length = len(trajectory_states)
            for position in range(trajectory_length - 1):
                if self.subgoal_distance_interval is None:
                    distances = range(1, trajectory_length - position)
                else:
                    distances = self.subgoal_distance_interval
                for dist in distances:
                    inner = min(int(dist), trajectory_length - 1 - position)
                    x, y = self._tokenize_pair(trajectory_states[position], trajectory_states[position + inner])
                    x_steps.append(x)
                    y_steps.append(y)
                    count += 1
                    if self.max_samples is not None and count >= self.max_samples:
                        break
                    if position + inner >= trajectory_length - 1:
                        break
                if self.max_samples is not None and count >= self.max_samples:
                    break
            if self.max_samples is not None and count >= self.max_samples:
                break

        if not x_steps:
            raise ValueError("No raw samples were tokenized")
        return torch.cat(x_steps, dim=0), torch.cat(y_steps, dim=0)

    def _tokenize_teacher(self) -> tuple[Tensor, Tensor]:
        x_steps: list[Tensor] = []
        y_steps: list[Tensor] = []
        for ann in tqdm(self._iter_teacher_annotations()):
            current_state = ann.get("current_state")
            proposition_state = ann.get("proposition_state")
            if current_state is None or proposition_state is None:
                continue
            x, y = self._tokenize_pair(current_state, proposition_state)
            x_steps.append(x)
            y_steps.append(y)

        if not x_steps:
            raise ValueError("No teacher samples were tokenized")
        return torch.cat(x_steps, dim=0), torch.cat(y_steps, dim=0)

    def prepare_data(self) -> None:
        logger.info(f"Preparing universal generator dataset for recipe={self.recipe}")
        if self.recipe == "raw":
            x_all, y_all = self._tokenize_raw()
        else:
            x_all, y_all = self._tokenize_teacher()

        if self.for_testing:
            joblib.dump(
                [x_all, y_all],
                self.save_tokenized_dataset_path / f"{self.env.name}_universal_{self.recipe}_tokenized_all_x_y",
            )
            return

        indices = np.arange(x_all.shape[0])
        train_idx, val_idx = train_test_split(indices, test_size=self.validation_split)
        train_x = x_all[train_idx]
        train_y = y_all[train_idx]
        val_x = x_all[val_idx]
        val_y = y_all[val_idx]

        logger.info(f"Universal train size ({self.recipe}): {len(train_x)}")
        logger.info(f"Universal val size ({self.recipe}): {len(val_x)}")

        joblib.dump(
            [train_x, train_y],
            self.save_tokenized_dataset_path / f"{self.env.name}_universal_{self.recipe}_tokenized_train_x_y",
        )
        joblib.dump(
            [val_x, val_y],
            self.save_tokenized_dataset_path / f"{self.env.name}_universal_{self.recipe}_tokenized_val_x_y",
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            x_train, y_train = joblib.load(
                self.save_tokenized_dataset_path / f"{self.env.name}_universal_{self.recipe}_tokenized_train_x_y"
            )
            self._train_dataset = GameDataset(x_train, y_train)

        if stage in ["fit", "validate"]:
            x_val, y_val = joblib.load(
                self.save_tokenized_dataset_path / f"{self.env.name}_universal_{self.recipe}_tokenized_val_x_y"
            )
            self._val_dataset = GameDataset(x_val, y_val)

        if stage in ["test"]:
            x_all, y_all = joblib.load(
                self.save_tokenized_dataset_path / f"{self.env.name}_universal_{self.recipe}_tokenized_all_x_y"
            )
            self._test_dataset = GameDataset(x_all, y_all)

    def get_train_dataset(self) -> Dataset:
        assert self._train_dataset is not None
        return self._train_dataset

    def get_val_dataset(self) -> Dataset:
        assert self._val_dataset is not None
        return self._val_dataset

    def get_test_dataset(self) -> Dataset:
        assert self._test_dataset is not None
        return self._test_dataset

    def collate_batch(self, xy: list[tuple[Tensor, Tensor]]) -> UniversalGeneratorBatch:
        input_ids = torch.stack([x for x, _ in xy], dim=0)
        labels = torch.stack([y for _, y in xy], dim=0)
        # Existing dataset stores rows without leading singleton dim after indexing.
        return UniversalGeneratorBatch.from_xy(input_ids, labels, recipe=self.recipe, pad_token_id=self.pad_token_id)

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_batch,
        )
