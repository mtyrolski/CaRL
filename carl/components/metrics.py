# pylint: disable=arguments-differ

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics import MetricCollection
from transformers import EvalPrediction


class GeneratorTokenAccuracy(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total

    @staticmethod
    def _input_format(logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        preds: Tensor = torch.argmax(logits, dim=-1)
        return preds, target


class GeneratorSequenceTokensAccuracy(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        agg: Tensor = (preds == target).all(axis=1)
        self.correct += torch.sum(agg)
        self.total += agg.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total

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

        if self.type_of_evaluation == 'classification':
            process_and_compute_metrics: tuple[
                Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                Callable[[EvalPrediction], dict],
            ] = self.preprocess_and_compute_value_classification_metrics()
        else:
            process_and_compute_metrics: tuple[None, None] = self.preprocess_and_compute_value_regression_metrics()

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
            l2_loss_expected_distance: np.ndarray = np.mean(np.square(expected_distance - target))
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
