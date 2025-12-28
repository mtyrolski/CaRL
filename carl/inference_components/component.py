import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias
from typing import TypeVar
from typing import Protocol
from typing import cast

from loguru import logger
import torch
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments
from carl.utils.training_metrics import MetricsHF
from carl.utils.retry import RetryConfig
from carl.utils.retry import retry

@dataclass
class TrainingModule:
    trainer_class: type[HFTrainer]
    trainer_args: Callable[..., TrainingArguments]
    metrics_for_component: MetricsHF

RawSimpleComponent: TypeAlias = torch.nn.Module
RawComplexComponent: TypeAlias = Mapping[int, torch.nn.Module] | Mapping[str, torch.nn.Module]

RawComponent: TypeAlias = RawSimpleComponent | RawComplexComponent

NetworkFromPath: TypeAlias = Callable[[str], RawSimpleComponent]

ComplexTrainingModule: TypeAlias = dict[str, TrainingModule]

_TModule = TypeVar("_TModule", bound=torch.nn.Module)
_TModule_co = TypeVar("_TModule_co", bound=torch.nn.Module, covariant=True)


class _FromPretrained(Protocol[_TModule_co]):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> _TModule_co: ...


_PATH_RESOLUTION_RETRY = RetryConfig(
    attempts=5,
    delay_seconds=10.0,
    backoff=1.0,
    retry_on=(FileNotFoundError, RuntimeError),
    on_retry=lambda attempt, exc: logger.info(
        f"Weights path not ready (attempt {attempt}). Retrying... ({exc})"
    ),
)

_LOAD_RETRY = RetryConfig(
    attempts=5,
    delay_seconds=10.0,
    backoff=1.0,
    retry_on=(Exception,),
    on_retry=lambda attempt, exc: logger.critical(
        f"Failed to load weights (attempt {attempt}). Retrying... ({exc})"
    ),
)


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

    def _resolve_weights_path(self, weights_path: str) -> str:
        if not weights_path:
            raise RuntimeError('Empty weights_path')

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'Path does not exist: {weights_path}')

        basename = os.path.basename(weights_path)
        if basename.startswith('checkpoint'):
            return weights_path

        if not os.path.isdir(weights_path):
            # Not a checkpoint dir and not a directory containing checkpoints.
            return weights_path

        entries = [f for f in os.listdir(weights_path) if f.startswith('checkpoint')]
        if len(entries) == 0:
            raise RuntimeError(f'No checkpoint* entries found in {weights_path}')
        if len(entries) > 1:
            raise RuntimeError(f'Found multiple checkpoint* entries in {weights_path}: {entries}')
        return os.path.join(weights_path, entries[0])

    def _validate_weights_path(self, resolved_path: str) -> None:
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f'Resolved weights path does not exist: {resolved_path}')

        if os.path.isdir(resolved_path):
            files = os.listdir(resolved_path)
            if len(files) == 0:
                raise RuntimeError(f'Resolved weights directory is empty: {resolved_path}')

            allowed_markers = (
                'config.json',
                'pytorch_model.bin',
                'model.safetensors',
            )
            allowed_suffixes = ('.safetensors', '.bin', '.pt', '.pth', '.ckpt')

            has_marker = any(name in files for name in allowed_markers)
            has_payload = any(name.endswith(allowed_suffixes) for name in files)
            if not (has_marker or has_payload):
                logger.warning(
                    f"Weights directory {resolved_path} doesn't look like a typical checkpoint. "
                    f"Proceeding anyway; loader may still handle it. Files: {files}"
                )
        else:
            if os.path.getsize(resolved_path) == 0:
                raise RuntimeError(f'Resolved weights file is empty: {resolved_path}')

    @retry(_PATH_RESOLUTION_RETRY)
    def _resolve_and_validate_weights_path(self, weights_path: str) -> str:
        resolved = self._resolve_weights_path(weights_path)
        self._validate_weights_path(resolved)
        return resolved

    @retry(_LOAD_RETRY)
    def _load_network_once(
        self,
        network_factory: Callable[[str], _TModule] | type[_TModule],
        resolved_path: str,
    ) -> _TModule:
        if isinstance(network_factory, type) and hasattr(network_factory, 'from_pretrained'):
            factory = cast(_FromPretrained[_TModule], network_factory)
            model = factory.from_pretrained(resolved_path)
        else:
            model = cast(Callable[[str], _TModule], network_factory)(resolved_path)
        logger.success(f'Loaded weights from {resolved_path}')
        return model

    def instantiate_network(
        self,
        network_factory: Callable[[str], _TModule] | type[_TModule],
        weights_path: str,
    ) -> _TModule:
        """Instantiate a network using a callable that accepts a weights path.

        Contract: `network_fn(resolved_weights_path)` must return a torch.nn.Module.
        This method then moves it to `self.device` and puts it in eval mode.
        """

        logger.debug(f'Loading weights from {weights_path}')
        resolved_path = self._resolve_and_validate_weights_path(weights_path)
        logger.info(f'Resolved weights path: {resolved_path}')

        model = self._load_network_once(network_factory, resolved_path)
        model.to(self.device)
        model.eval()
        return model

    def is_trainable(self) -> bool:
        """Returns whether the component is trainable."""

        return self.get_component_training_module() is not None
