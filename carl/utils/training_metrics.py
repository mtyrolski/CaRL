from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from functools import reduce
from operator import or_
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics import MetricCollection
from transformers import EvalPrediction

from carl.planners.base import Experience

Metrics = dict[str, float | list[float]]


def is_safe_list(lst: list[Any]) -> bool:
    return all(isinstance(x, (int, float)) for x in lst)


def safe_metric_update(logs, key, value) -> bool:
    if value is None:
        return False

    if isinstance(value, (int, float)):
        logs[key].append(value)

    elif isinstance(value, list):
        if is_safe_list(value):
            logs[key].extend(value)
            return True
        return False

    elif isinstance(value, bool):
        logs[key].append(int(value))
        return True

    return False


def extract_metrics_from_experiences(experiences: list[Experience]) -> Metrics:
    keys_solution = reduce(or_, (experience.solution.__dict__.keys() for experience in experiences))
    keys_search_info = reduce(or_, (experience.search_info.__dict__.keys() for experience in experiences))

    logs: Metrics = {k: [] for k in keys_solution | keys_search_info}

    trackable_metrics: int = 0
    non_trackable_metrics: int = 0

    for experience in experiences:
        solution = experience.solution
        search_info = experience.search_info
        for key, value in solution.__dict__.items():
            trackable_metrics += int(safe_metric_update(logs, key, value))
            non_trackable_metrics += int(not safe_metric_update(logs, key, value))

        for key, value in search_info.__dict__.items():
            trackable_metrics += int(safe_metric_update(logs, key, value))
            non_trackable_metrics += int(not safe_metric_update(logs, key, value))

    return logs


def extract_metrics_from_buffer_logs(buffer_logs: list[dict[str, float | int]]) -> Metrics:
    # For each log sum and average the values
    keys = buffer_logs[0].keys()
    assert all(log.keys() == keys for log in buffer_logs)

    logs_sum_within_iteration: dict[str, float] = {f'{key}/sum': sum(log[key] for log in buffer_logs) for key in keys}

    logs_avg_within_iteration: dict[str, float] = {
        f'{key}/avg': logs_sum_within_iteration[f'{key}/sum'] / len(buffer_logs) for key in keys
    }

    logs_low_level: dict[str, list[float]] = {key: [] for key in keys}

    for log in buffer_logs:
        for key in keys:
            logs_low_level[key].append(log[key])

    return {**logs_sum_within_iteration, **logs_avg_within_iteration, **logs_low_level}


class GeneratorTokenAccuracy(Metric):
    correct: Tensor
    total: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        correct: Tensor = cast(Tensor, self.correct)
        total: Tensor = cast(Tensor, self.total)
        correct = correct + torch.sum(preds == target)
        total = total + target.numel()
        self.correct = correct
        self.total = total

    def compute(self) -> Tensor:
        correct: Tensor = cast(Tensor, self.correct)
        total: Tensor = cast(Tensor, self.total)
        return correct.float() / total

    @staticmethod
    def _input_format(logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        preds: Tensor = torch.argmax(logits, dim=-1)
        return preds, target


class GeneratorSequenceTokensAccuracy(Metric):
    correct: Tensor
    total: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        agg: Tensor = (preds == target).all(dim=1)
        correct = cast(Tensor, self.correct)
        total = cast(Tensor, self.total)
        correct = correct + torch.sum(agg)
        total = total + agg.numel()
        self.correct = correct
        self.total = total

    def compute(self) -> Tensor:
        correct = cast(Tensor, self.correct)
        total = cast(Tensor, self.total)
        return correct.float() / total

    @staticmethod
    def _input_format(logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        preds: Tensor = torch.argmax(logits, dim=-1)
        return preds, target


class GeneratorAccuracy:
    def __init__(self) -> None:
        self.token_accuracy: Metric = GeneratorTokenAccuracy()
        self.sequence_accuracy: Metric = GeneratorSequenceTokensAccuracy()

    def combined_metrics(self) -> MetricCollection:
        return MetricCollection([self.token_accuracy, self.sequence_accuracy])


class MetricsHF(ABC):
    """
    Abstract class for metrics.

    This class is used to define the metrics that will be used during training by the
    HuggingFace Trainer.
    """
    @abstractmethod
    def get_metrics(
        self,
    ) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]] | tuple[None,
                                                                                                               None]:
        raise NotImplementedError


class ValueMetricsHF(MetricsHF):
    """
    Metrics for the value.

    This class is used to define the metrics that will
    be used during training by the HuggingFace Trainer. Form more information how HF's Trainer handles inputs
    preprocessing and metrics, see https://huggingface.co/docs/transformers/main_classes/trainer
    """
    def __init__(self, type_of_evaluation: str) -> None:
        self.type_of_evaluation: str = type_of_evaluation
        assert self.type_of_evaluation in [
            'classification',
            'regression',
        ], f'Invalid type of evaluation: {self.type_of_evaluation}'

    def get_metrics(
        self,
    ) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]] | tuple[None,
                                                                                                               None]:
        process_and_compute_metrics: (
            tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]]
            | tuple[None, None]
        )
        if self.type_of_evaluation == 'classification':
            process_and_compute_metrics = self.preprocess_and_compute_value_classification_metrics()
        else:
            process_and_compute_metrics = self.preprocess_and_compute_value_regression_metrics()

        return process_and_compute_metrics

    @staticmethod
    def preprocess_and_compute_value_classification_metrics(
    ) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]]:
        def preprocess_logits_for_metrics(logits: Tensor, labels: Tensor) -> Tensor:    # type: ignore
            probs: Tensor = torch.tensor(logits.softmax(dim=-1))
            return probs

        def value_metrics(eval_preds: EvalPrediction) -> dict:
            probs: np.ndarray = eval_preds[0]
            target: np.ndarray = eval_preds[1]
            preds: np.ndarray = np.argmax(probs, axis=-1)
            assert preds.shape == target.shape
            distances: np.ndarray = np.arange(0, len(probs[0]))
            expected_distance: np.ndarray = np.array([np.inner(probs[i], distances) for i in range(len(probs))])
            l2_loss_expected_distance: float = float(np.mean(np.square(expected_distance - target)))
            return {
                'value_accuracy': (preds == target).astype(float).mean().item(),
                'l2_loss_expected_distance': l2_loss_expected_distance,
            }

        return preprocess_logits_for_metrics, value_metrics

    @staticmethod
    def preprocess_and_compute_value_regression_metrics() -> tuple[None, None]:
        return None, None


class PolicyMetricsHF(MetricsHF):
    """
    Metrics for the policy.

    This class is used to define the metrics that will
    be used during training by the HuggingFace Trainer. Form more information how HF's Trainer handles inputs
    preprocessing and metrics, see https://huggingface.co/docs/transformers/main_classes/trainer
    """
    def get_metrics(
        self,) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]]:
        return self.preprocess_and_compute_policy_metrics()

    @staticmethod
    def preprocess_and_compute_policy_metrics(
    ) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]]:
        def preprocess_logits_for_metrics(logits: Tensor, labels: Tensor) -> Tensor:    # type: ignore
            pred_ids: Tensor = logits.argmax(dim=-1)
            return pred_ids

        def policy_metrics(eval_preds: EvalPrediction) -> dict:
            preds: np.ndarray = eval_preds[0]
            target: np.ndarray = eval_preds[1]
            assert preds.shape == target.shape
            return {'policy_accuracy': (preds == target).astype(float).mean().item()}

        return preprocess_logits_for_metrics, policy_metrics


class ConditionalLowLevelPolicyMetricsHF(MetricsHF):
    """
    Metrics for the conditional low level policy.

    This class is used to define the metrics that will
    be used during training by the HuggingFace Trainer. Form more information how HF's Trainer handles inputs
    preprocessing and metrics, see https://huggingface.co/docs/transformers/main_classes/trainer
    """
    def get_metrics(
        self,) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]]:
        return self.preprocess_and_compute_cllp_metrics()

    @staticmethod
    def preprocess_and_compute_cllp_metrics(
    ) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]]:
        def preprocess_logits_for_metrics(logits: Tensor, labels: Tensor) -> Tensor:    # type: ignore
            pred_ids: Tensor = torch.argmax(logits, dim=-1)
            return pred_ids

        def cllp_metrics(eval_preds: EvalPrediction) -> dict:
            preds: np.ndarray = eval_preds[0]
            target: np.ndarray = eval_preds[1]
            assert preds.shape == target.shape
            return {'cllp_accuracy': (preds == target).astype(float).mean().item()}

        return preprocess_logits_for_metrics, cllp_metrics


class GeneratorMetricsHF(MetricsHF):
    """
    Metrics for the conditional low level policy.

    This class is used to define the metrics that will
    be used during training by the HuggingFace Trainer. Form more information how HF's Trainer handles inputs
    preprocessing and metrics, see https://huggingface.co/docs/transformers/main_classes/trainer
    """
    def get_metrics(
        self,) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]]:
        return self.preprocess_and_compute_generator_metrics()

    @staticmethod
    def preprocess_and_compute_generator_metrics(
    ) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]]:
        def preprocess_logits_for_metrics(logits: Tensor, labels: Tensor) -> Tensor:    # type: ignore
            pred_ids: Tensor = logits[0].argmax(dim=-1)
            return pred_ids

        def generator_metrics(eval_preds: EvalPrediction) -> dict:
            preds: np.ndarray = eval_preds[0]
            target: np.ndarray = eval_preds[1]
            assert preds.shape == target.shape
            return {
                'tokens_accuracy': (preds == target).astype(float).mean().item(),
                'tokens_sequence_accuracy': (preds == target).all(axis=1).astype(float).mean().item(),
            }

        return preprocess_logits_for_metrics, generator_metrics
