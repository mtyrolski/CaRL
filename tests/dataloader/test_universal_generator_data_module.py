from __future__ import annotations

from pathlib import Path

import joblib
import torch

from carl.dataloader.universal_generator_data_module import UniversalGeneratorDataModule
from carl.dataloader.universal_generator_types import UniversalGeneratorBatch


def _to_int(value) -> int:
    if isinstance(value, str):
        return int(value)
    return int(value)


class TinySeqTokenizer:
    def x_y_tokenizer(self, x, y, training_goal):  # noqa: ARG002
        x_i = _to_int(x)
        y_i = _to_int(y)
        # fixed-length token sequences compatible with seq2seq models
        return (
            torch.tensor([[6, x_i + 7, 3, 2]], dtype=torch.long),
            torch.tensor([[y_i + 7, 3, 2, 1]], dtype=torch.long),
        )


class TinySeqEnv:
    name = "tinyseq"

    def __init__(self) -> None:
        self.tokenizer = TinySeqTokenizer()


def _raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    joblib.dump({0: ["0", "1", "2", "3"], 1: ["2", "3", "4"]}, raw_dir / "part.joblib")
    return raw_dir


def _teacher_file(tmp_path: Path) -> Path:
    teacher = tmp_path / "teacher.joblib"
    payload = {
        "meta": {"source": "test"},
        "annotations": [
            {
                "current_state": "0",
                "proposition_state": "2",
                "validator_accept": True,
                "validator_reject": False,
                "reached": True,
                "source": "test",
            },
            {
                "current_state": "1",
                "proposition_state": "3",
                "validator_accept": True,
                "validator_reject": False,
                "reached": False,
                "source": "test",
            },
            {
                "current_state": "2",
                "proposition_state": "4",
                "validator_accept": True,
                "validator_reject": False,
                "reached": True,
                "source": "test",
            },
            {
                "current_state": "3",
                "proposition_state": "5",
                "validator_accept": False,
                "validator_reject": True,
                "reached": False,
                "source": "test",
            },
        ],
    }
    joblib.dump(payload, teacher)
    return teacher


def _assert_batch_shapes(batch: UniversalGeneratorBatch, recipe: str) -> None:
    assert isinstance(batch, UniversalGeneratorBatch)
    assert batch.recipe == recipe
    assert batch.input_ids.ndim == 2
    assert batch.labels.ndim == 2
    assert batch.attention_mask.shape == batch.input_ids.shape
    assert batch.labels_attention_mask.shape == batch.labels.shape
    print(recipe, "input_ids", tuple(batch.input_ids.shape), "labels", tuple(batch.labels.shape))


def test_universal_datamodule_raw_finetune_contrastive_smoke(tmp_path: Path):
    env = TinySeqEnv()
    raw_dir = _raw_dir(tmp_path)
    teacher_file = _teacher_file(tmp_path)

    dm_raw = UniversalGeneratorDataModule(
        env=env,
        recipe="raw",
        save_tokenized_dataset_path=str(tmp_path / "tok_raw"),
        raw_dataset_path=raw_dir,
        subgoal_distance_interval=[2],
        validation_split=0.5,
        batch_size=2,
        num_workers=0,
    )
    dm_raw.prepare_data()
    dm_raw.setup("fit")
    raw_batch = next(iter(dm_raw.train_dataloader()))
    _assert_batch_shapes(raw_batch, "raw")

    dm_ft = UniversalGeneratorDataModule(
        env=env,
        recipe="finetune",
        save_tokenized_dataset_path=str(tmp_path / "tok_ft"),
        teacher_dataset_path=teacher_file,
        teacher_accept_only=True,
        validation_split=0.5,
        batch_size=2,
        num_workers=0,
    )
    dm_ft.prepare_data()
    dm_ft.setup("fit")
    ft_batch = next(iter(dm_ft.train_dataloader()))
    _assert_batch_shapes(ft_batch, "finetune")

    dm_con = UniversalGeneratorDataModule(
        env=env,
        recipe="contrastive",
        save_tokenized_dataset_path=str(tmp_path / "tok_con"),
        teacher_dataset_path=teacher_file,
        teacher_accept_only=True,
        validation_split=0.5,
        batch_size=2,
        num_workers=0,
    )
    dm_con.prepare_data()
    dm_con.setup("fit")
    con_batch = next(iter(dm_con.train_dataloader()))
    _assert_batch_shapes(con_batch, "contrastive")
