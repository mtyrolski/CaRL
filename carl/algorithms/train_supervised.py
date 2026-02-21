from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from joblib import load
from lightning import LightningDataModule
from lightning import LightningModule
from lightning import Trainer
from torch.utils.data import Dataset
from transformers import EvalPrediction
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers import Trainer as HFTrainer
from transformers import TrainerCallback
from loguru import logger

from carl.algorithms.algorithm import Algorithm
from carl.dataloader.game_data_module import GameDataModule
from carl.utils.loggers import CaRLLogger
from carl.utils.training_metrics import MetricsHF


class TrainSupervised(Algorithm):
    """Train a model using the Lightning Trainer."""
    def __init__(
        self,
        component: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        custom_logger: CaRLLogger | None = None,
    ) -> None:
        super().__init__()
        self.component = component
        self.datamodule = datamodule
        self.trainer = trainer
        self.custom_logger = custom_logger

    def run(self) -> None:
        logger.info('Starting training')
        self.trainer.fit(self.component, self.datamodule)
        # Consider adding a callback for component evaluation here.


class TrainSupervisedHF(Algorithm):
    """Train a model using the HuggingFace Trainer."""
    def __init__(
        self,
        trainer: Callable[..., HFTrainer],
        model: Callable[..., PreTrainedModel] | type[PreTrainedModel],
        datamodule: GameDataModule,
        custom_logger: CaRLLogger | None = None,
        cllp_logger: Callable[..., TrainerCallback] | None = None,
        path_to_data_to_test_cllp: str | None = None,
        custom_metrics: MetricsHF | None = None,
        config: PretrainedConfig | None = None,
        do_finetune: bool = False,
        path_to_model_weights: str | None = None,
    ) -> None:
        super().__init__()

        self.datamodule = datamodule
        self.custom_logger = custom_logger
        self.cllp_logger = cllp_logger
        self.path_to_data_to_test_cllp = path_to_data_to_test_cllp
        self.custom_metrics = custom_metrics

        self.model_to_train: PreTrainedModel | None = None
        self.ready_trainer: HFTrainer | None = None

        if not do_finetune:
            assert config is not None, 'config must be provided if do_finetune is False'
            self.config = config
            self.model_to_train = model(config=self.config)
            logger.success('Instantiated raw model from config')
        else:
            assert (path_to_model_weights is not None), 'path_to_model_weights must be provided if do_finetune is True'
            assert hasattr(model, "from_pretrained")
            model_cls = cast(type[PreTrainedModel], model)
            self.model_to_train = model_cls.from_pretrained(path_to_model_weights)
            logger.success('Loaded model checkpoint (arch+weights) over the config')

        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        compute_metrics: Callable[[EvalPrediction], dict] | None

        if self.custom_metrics is not None:
            preprocess_logits_for_metrics, compute_metrics = self.custom_metrics.get_metrics()
        else:
            preprocess_logits_for_metrics = None
            compute_metrics = None
        logger.info('Setting up datasets')
        self.datamodule.prepare_data()
        self.datamodule.setup('fit')
        train_dataset: Dataset = self.datamodule.get_train_dataset()
        validation_dataset: Dataset = self.datamodule.get_val_dataset()
        logger.debug(f'Datasets are ready, train: {train_dataset}, validation: {validation_dataset}')
        self.ready_trainer = trainer(
            model=self.model_to_train,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            data_collator=self.data_collector,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
        )
        logger.success('Trainer is ready')

        if self.custom_logger is not None:
            trainer_logger = self.custom_logger.return_logger()
            self.ready_trainer.add_callback(trainer_logger)

            if self.cllp_logger is not None and self.datamodule.training_goal.value == 'cllp':
                assert (self.path_to_data_to_test_cllp
                        is not None), 'path_to_data_to_test_cllp must be provided if cllp_logger is not None'
                data_to_test_cllp: dict[int, list[np.ndarray]] = load(self.path_to_data_to_test_cllp)
                cllp_callback: TrainerCallback = self.cllp_logger(
                    inner_logger=trainer_logger,
                    data_to_evaluate=data_to_test_cllp,
                    distance_range=self.datamodule.subgoal_distance_interval,
                    env=self.datamodule.env,
                )
                self.ready_trainer.add_callback(cllp_callback)

    @staticmethod
    def data_collector(xy: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            'input_ids': torch.stack([x[0] for x in xy]),
            'labels': torch.stack([y[1] for y in xy]),
        }

    def run(self) -> None:
        logger.info('Training model')
        assert self.ready_trainer is not None
        self.ready_trainer.train()
