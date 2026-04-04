from __future__ import annotations

from pathlib import Path

import joblib
import torch
from transformers import BartConfig
from transformers import BartForConditionalGeneration

from carl.algorithms.train_universal_generator import TrainUniversalGeneratorHF
from carl.dataloader.universal_generator_data_module import UniversalGeneratorDataModule


def _to_int(value) -> int:
    return int(value) if isinstance(value, str) else int(value)


class TinySeqTokenizer:
    def x_y_tokenizer(self, x, y, training_goal):  # noqa: ARG002
        x_i = _to_int(x)
        y_i = _to_int(y)
        return (
            torch.tensor([[6, x_i + 7, 3, 2]], dtype=torch.long),
            torch.tensor([[y_i + 7, 3, 2]], dtype=torch.long),
        )


class TinySeqEnv:
    name = "tinyseq"

    def __init__(self) -> None:
        self.tokenizer = TinySeqTokenizer()


def _write_raw_dataset(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            0: ["0", "1", "2", "3", "4"],
            1: ["1", "2", "3", "4", "5"],
            2: ["2", "3", "4", "5", "6"],
        },
        raw_dir / "part.joblib",
    )
    return raw_dir


def _write_teacher_dataset(tmp_path: Path) -> Path:
    teacher_file = tmp_path / "teacher.joblib"
    teacher_file.parent.mkdir(parents=True, exist_ok=True)
    anns = []
    for i in range(8):
        anns.append(
            {
                "current_state": str(i),
                "proposition_state": str(i + 2),
                "validator_accept": True,
                "validator_reject": False,
                "reached": i % 2 == 0,
                "source": "test",
            }
        )
    joblib.dump({"meta": {}, "annotations": anns}, teacher_file)
    return teacher_file


def _tiny_bart_config() -> BartConfig:
    return BartConfig(
        vocab_size=32,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=32,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=0,
    )


def _run_recipe(tmp_path: Path, recipe: str, init_ckpt: str | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = TinySeqEnv()
    raw_dir = _write_raw_dataset(tmp_path)
    teacher_file = _write_teacher_dataset(tmp_path)
    dm_kwargs = {
        "env": env,
        "recipe": recipe,
        "save_tokenized_dataset_path": str(tmp_path / f"tok_{recipe}"),
        "validation_split": 0.5,
        "batch_size": 2,
        "num_workers": 0,
    }
    if recipe == "raw":
        dm_kwargs.update({"raw_dataset_path": raw_dir, "subgoal_distance_interval": [2], "max_samples": 32})
    else:
        dm_kwargs.update({"teacher_dataset_path": teacher_file, "teacher_accept_only": True, "max_samples": 32})

    dm = UniversalGeneratorDataModule(**dm_kwargs)  # type: ignore[arg-type]
    output_dir = tmp_path / f"out_{recipe}"
    cfg = _tiny_bart_config()

    trainer = TrainUniversalGeneratorHF(
        model=BartForConditionalGeneration,
        datamodule=dm,
        output_dir=str(output_dir),
        recipe=recipe,
        config=None if recipe != "raw" else cfg,
        do_finetune=recipe != "raw",
        path_to_model_weights=init_ckpt,
        learning_rate=1e-3,
        max_steps=2,
        num_train_epochs=2,
        eval_every_n_steps=0,
        log_every_n_steps=1,
        save_every_n_steps=0,
        lambda_raw=1.0 if recipe == "raw" else 0.0,
        lambda_imit=1.0 if recipe in {"finetune", "contrastive"} else 0.0,
        lambda_contrastive=0.2 if recipe == "contrastive" else 0.0,
        temperature=0.2,
        prefer_cuda=False,
    )
    trainer.run()

    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    assert checkpoints, f"No checkpoint written for recipe={recipe}"
    assert (output_dir / "training_summary.json").exists()
    return str(checkpoints[-1])


def test_train_universal_generator_smoke_raw_finetune_contrastive(tmp_path: Path):
    init_cfg = _tiny_bart_config()
    init_model = BartForConditionalGeneration(init_cfg)
    init_ckpt_dir = tmp_path / "init_ckpt"
    init_model.save_pretrained(init_ckpt_dir)

    _run_recipe(tmp_path / "raw_case", "raw", None)
    _run_recipe(tmp_path / "ft_case", "finetune", str(init_ckpt_dir))
    _run_recipe(tmp_path / "con_case", "contrastive", str(init_ckpt_dir))
