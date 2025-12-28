import os
import time
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger
import torch
from transformers import PreTrainedModel
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments
from typing import TypeAlias
from carl.utils.training_metrics import MetricsHF

@dataclass
class TrainingModule:
    trainer_class: type[HFTrainer]
    trainer_args: Callable[..., TrainingArguments]
    metrics_for_component: MetricsHF

RawSimpleComponent: TypeAlias = PreTrainedModel
RawComplexComponent: TypeAlias = dict[int, PreTrainedModel] | dict[str, PreTrainedModel]

RawComponent: TypeAlias = RawSimpleComponent | RawComplexComponent

ComplexTrainingModule: TypeAlias = dict[str, TrainingModule]


class InferenceComponent(ABC):
    device: torch.device
    
    @abstractmethod
    def get_network(self) -> RawComponent:
        """
        Returns the networks.

        Dict for nested inference components which consist of multiple networks.
        (such as adaptive subgoal generator).
        """

        raise NotImplementedError

    @abstractmethod
    def construct_network(self) -> None:
        """Construct the networks."""

        raise NotImplementedError

    def get_component_training_module(self) -> TrainingModule | ComplexTrainingModule | None:
        """
        Returns the training module.

        Dict for nested inference components which consist of multiple networks.
        (such as adaptive subgoal generator).
        """
        return None

    def instantiate_network(self, network_fn, weights_path: str):
        """Instantiate the networks."""
        logger.debug(f'Loading weights from {weights_path}')
        full_weights_path = None
        for _ in range(5):
            if weights_path is not None and os.path.exists(weights_path):
                basename = os.path.basename(weights_path)

                if not basename.startswith('checkpoint'):
                    fs = os.listdir(weights_path)
                    fs = [f for f in fs if f.startswith('checkpoint')]

                    if len(fs) == 0:
                        logger.info(
                            f'Checkpoints has not been found in {weights_path}. Waiting 10 seconds and retrying...')
                        time.sleep(10)
                        continue

                    assert len(fs) == 1, f'Found multiple checkpoints in {weights_path}'

                    ckpt_folder_name = fs[0]
                    full_weights_path = os.path.join(weights_path, ckpt_folder_name)
                    break
                else:
                    logger.info('Provided direct path to the ckpt')
                    full_weights_path = weights_path
                    break
            else:
                logger.info(f'Path {weights_path} does not exist.')
                time.sleep(10)
                continue

        network = None
        for _ in range(5):
            try:
                network = network_fn(full_weights_path)
                logger.success(f'Loaded weights from {full_weights_path}')
                break
            except Exception as e:
                logger.critical(f'Failed to load weights from {full_weights_path}. Retrying...')
                logger.critical(e)
                time.sleep(10)
        if network is None:
            raise RuntimeError(f'Failed to load weights from {full_weights_path}')

        network.to(self.device)
        network.eval()

        return network

    def is_trainable(self) -> bool:
        """Returns whether the component is trainable."""

        return self.get_component_training_module() is not None
