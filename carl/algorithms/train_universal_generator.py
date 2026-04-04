from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch
from loguru import logger
from transformers import PreTrainedModel
from transformers import PretrainedConfig

from carl.algorithms.algorithm import Algorithm
from carl.dataloader.universal_generator_data_module import UniversalGeneratorDataModule
from carl.dataloader.universal_generator_types import UniversalGeneratorBatch
from carl.utils.loggers import CaRLLogger
from carl.utils.torch_device import resolve_device
from carl.utils.universal_generator_losses import in_batch_infonce
from carl.utils.universal_generator_losses import masked_mean_pool
from carl.utils.universal_generator_losses import verifier_consistency_placeholder


class TrainUniversalGeneratorHF(Algorithm):
    """Recipe-aware trainer for universal propositional generator experiments.

    Reuses the same seq2seq architecture and tokenizer IO as the existing generator
    (`state -> proposition state`) and adds an optional InfoNCE loss for the
    `contrastive` recipe.
    """

    def __init__(
        self,
        model: Callable[..., PreTrainedModel] | type[PreTrainedModel],
        datamodule: UniversalGeneratorDataModule,
        output_dir: str,
        *,
        recipe: str,
        config: PretrainedConfig | None = None,
        do_finetune: bool = False,
        path_to_model_weights: str | None = None,
        custom_logger: CaRLLogger | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        max_steps: int = 1000,
        num_train_epochs: int = 1,
        eval_every_n_steps: int = 0,
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 0,
        grad_clip_norm: float = 1.0,
        temperature: float = 0.1,
        lambda_raw: float = 1.0,
        lambda_imit: float = 1.0,
        lambda_contrastive: float = 1.0,
        lambda_verifier: float = 0.0,
        prefer_cuda: bool = True,
    ) -> None:
        if recipe not in {"raw", "finetune", "contrastive"}:
            raise ValueError(f"Unsupported recipe: {recipe}")
        self.recipe = recipe
        self.datamodule = datamodule
        self.custom_logger = custom_logger
        self.output_dir = Path(output_dir)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.eval_every_n_steps = eval_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        self.grad_clip_norm = grad_clip_norm
        self.temperature = temperature
        self.lambda_raw = lambda_raw
        self.lambda_imit = lambda_imit
        self.lambda_contrastive = lambda_contrastive
        self.lambda_verifier = lambda_verifier
        self.device = resolve_device(prefer_cuda=prefer_cuda)

        if not do_finetune:
            if config is None:
                raise ValueError("config must be provided when do_finetune=False")
            self.model = model(config=config)
            logger.success("Instantiated universal generator model from config")
        else:
            if path_to_model_weights is None:
                raise ValueError("path_to_model_weights must be provided when do_finetune=True")
            if hasattr(model, "from_pretrained"):
                model_cls = cast(type[PreTrainedModel], model)
                self.model = model_cls.from_pretrained(path_to_model_weights)
            else:
                model_factory = cast(Callable[[str], PreTrainedModel], model)
                self.model = model_factory(path_to_model_weights)
            logger.success(f"Loaded universal generator init checkpoint from {path_to_model_weights}")

        self.model = cast(PreTrainedModel, self.model)
        self.model.to(self.device)

    def _neptune_log(self, key: str, step: int, value: float) -> None:
        if self.custom_logger is None:
            return
        run = getattr(self.custom_logger, "run", None)
        if run is None:
            return
        try:
            run[key].append(step=step, value=value)
        except Exception:
            logger.debug(f"Failed to log {key} to custom logger")

    def _encode_embeddings(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_getter = getattr(self.model, "get_encoder", None)
        if callable(encoder_getter):
            encoder = encoder_getter()
            encoder_outputs = encoder(input_ids=token_ids, attention_mask=attention_mask)
            hidden_states = encoder_outputs.last_hidden_state
            return masked_mean_pool(hidden_states, attention_mask)

        embedding_layer = self.model.get_input_embeddings()
        hidden_states = embedding_layer(token_ids)
        return masked_mean_pool(hidden_states, attention_mask)

    def _compute_losses(self, batch: UniversalGeneratorBatch) -> dict[str, torch.Tensor]:
        hf_batch = batch.to_hf_dict()
        hf_batch = {k: v.to(self.device) for k, v in hf_batch.items()}
        labels = hf_batch["labels"]
        attention_mask = batch.attention_mask.to(self.device)
        labels_attention_mask = batch.labels_attention_mask.to(self.device)

        outputs = self.model(**hf_batch)
        seq_loss = outputs.loss
        if seq_loss is None:
            raise RuntimeError("Model did not return a loss")

        zero = torch.zeros((), device=self.device)
        l_raw = zero
        l_imit = zero
        l_contrastive = zero
        l_verifier = verifier_consistency_placeholder(self.device)

        if self.recipe == "raw":
            l_raw = seq_loss
        elif self.recipe == "finetune":
            l_imit = seq_loss
        elif self.recipe == "contrastive":
            l_imit = seq_loss
            if labels.shape[0] > 1:
                anchor_embeddings = self._encode_embeddings(hf_batch["input_ids"], attention_mask)
                positive_embeddings = self._encode_embeddings(labels, labels_attention_mask)
                l_contrastive = in_batch_infonce(anchor_embeddings, positive_embeddings, temperature=self.temperature)

        total = (
            self.lambda_raw * l_raw
            + self.lambda_imit * l_imit
            + self.lambda_contrastive * l_contrastive
            + self.lambda_verifier * l_verifier
        )
        return {
            "loss": total,
            "loss_raw": l_raw.detach(),
            "loss_imit": l_imit.detach(),
            "loss_contrastive": l_contrastive.detach(),
            "loss_verifier": l_verifier.detach(),
            "loss_seq": seq_loss.detach(),
        }

    @staticmethod
    def _batch_to_device(batch: UniversalGeneratorBatch, device: torch.device) -> UniversalGeneratorBatch:
        return UniversalGeneratorBatch(
            input_ids=batch.input_ids.to(device),
            labels=batch.labels.to(device),
            attention_mask=batch.attention_mask.to(device),
            labels_attention_mask=batch.labels_attention_mask.to(device),
            recipe=batch.recipe,
        )

    def _evaluate(self, step: int) -> dict[str, float]:
        self.model.eval()
        val_loader = self.datamodule.val_dataloader()
        agg: dict[str, float] = {
            "loss": 0.0,
            "loss_raw": 0.0,
            "loss_imit": 0.0,
            "loss_contrastive": 0.0,
            "loss_verifier": 0.0,
            "loss_seq": 0.0,
        }
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                assert isinstance(batch, UniversalGeneratorBatch)
                batch = self._batch_to_device(batch, self.device)
                losses = self._compute_losses(batch)
                for key in agg:
                    agg[key] += float(losses[key].item())
                n_batches += 1

        if n_batches > 0:
            for key in agg:
                agg[key] /= n_batches
                self._neptune_log(f"universal_eval/{key}", step, agg[key])
        self.model.train()
        return agg

    def _save_checkpoint(self, step: int) -> Path:
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        logger.success(f"Saved universal generator checkpoint to {ckpt_dir}")
        return ckpt_dir

    def run(self) -> None:
        logger.info(f"Preparing datamodule for recipe={self.recipe}")
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

        train_loader = self.datamodule.train_dataloader()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.train()
        global_step = 0
        train_started = perf_counter()

        for epoch in range(self.num_train_epochs):
            logger.info(f"Universal training epoch {epoch + 1}/{self.num_train_epochs} ({self.recipe})")
            for batch in train_loader:
                if global_step >= self.max_steps:
                    break
                assert isinstance(batch, UniversalGeneratorBatch)
                batch = self._batch_to_device(batch, self.device)

                optimizer.zero_grad(set_to_none=True)
                losses = self._compute_losses(batch)
                losses["loss"].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()
                global_step += 1

                if self.log_every_n_steps > 0 and global_step % self.log_every_n_steps == 0:
                    msg = {k: float(v.item()) for k, v in losses.items()}
                    logger.info(f"[universal/{self.recipe}] step={global_step} {msg}")
                    for key, value in msg.items():
                        self._neptune_log(f"universal_train/{key}", global_step, value)

                if self.eval_every_n_steps > 0 and global_step % self.eval_every_n_steps == 0:
                    eval_metrics = self._evaluate(global_step)
                    logger.info(f"[universal/{self.recipe}] eval@{global_step} {eval_metrics}")

                if self.save_every_n_steps > 0 and global_step % self.save_every_n_steps == 0:
                    self._save_checkpoint(global_step)

            if global_step >= self.max_steps:
                break

        final_ckpt = self._save_checkpoint(global_step)
        elapsed = perf_counter() - train_started
        summary: dict[str, Any] = {
            "recipe": self.recipe,
            "global_step": global_step,
            "elapsed_seconds": elapsed,
            "output_checkpoint": str(final_ckpt),
        }
        try:
            summary["final_eval"] = self._evaluate(global_step)
        except Exception as exc:
            summary["final_eval_error"] = str(exc)
            logger.warning(f"Final evaluation failed: {exc}")
        (self.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        logger.success(f"Universal training finished for recipe={self.recipe} in {elapsed:.2f}s")
