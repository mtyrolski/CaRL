from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from functools import reduce
from operator import or_
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics import MetricCollection
from transformers import EvalPrediction

Metrics = dict[str, float | list[float]]
from carl.planners.base import Experience


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
    SOLUTION_IDX = 0
    SEARCH_INFO_IDX = 1

    keys_solution = reduce(or_, (experience[SOLUTION_IDX].keys() for experience in experiences))
    keys_search_info = reduce(or_, (experience[SEARCH_INFO_IDX].keys() for experience in experiences))

    logs: Metrics = {k: [] for k in keys_solution | keys_search_info}

    trackable_metrics: int = 0
    non_trackable_metrics: int = 0

    for experience in experiences:
        solution, search_info = experience
        for key, value in solution.items():
            trackable_metrics += int(safe_metric_update(logs, key, value))
            non_trackable_metrics += int(not safe_metric_update(logs, key, value))

        for key, value in search_info.items():
            trackable_metrics += int(safe_metric_update(logs, key, value))
            non_trackable_metrics += int(not safe_metric_update(logs, key, value))

    return logs


def extract_metrics_from_buffer_logs(buffer_logs: list[dict[str, float]]) -> Metrics:
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


class StateEmbeddingAEMetricsHF(MetricsHF):
    """Metrics for autoencoder training."""
    
    def get_metrics(self) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]] | tuple[None, None]:
        def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
            """Convert logits to predictions for autoencoder."""
            # For autoencoder, logits are already the reconstructed states
            return logits.detach().cpu().numpy()

        def ae_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
            """Compute reconstruction metrics for autoencoder."""
            preds, labels = eval_preds
            
            # Compute MSE reconstruction loss
            mse_loss = np.mean((preds - labels) ** 2)
            
            # Compute MAE reconstruction loss
            mae_loss = np.mean(np.abs(preds - labels))
            
            # Compute R-squared (coefficient of determination)
            ss_res = np.sum((labels - preds) ** 2)
            ss_tot = np.sum((labels - labels.mean()) ** 2)
            r2_score = 1 - (ss_res / (ss_tot + 1e-8))
            
            return {
                'reconstruction_mse': mse_loss,
                'reconstruction_mae': mae_loss,
                'reconstruction_r2': r2_score,
            }

        return preprocess_logits_for_metrics, ae_metrics


class StateEmbeddingVAEMetricsHF(MetricsHF):
    """Metrics for VAE training."""
    
    def get_metrics(self) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]] | tuple[None, None]:
        def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
            """Convert VAE outputs to predictions."""
            # Assume logits contain reconstruction in first channels
            if isinstance(logits, dict):
                reconstruction = logits.get('reconstruction', logits.get('logits', logits))
            else:
                reconstruction = logits
            return reconstruction.detach().cpu().numpy()

        def vae_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
            """Compute reconstruction and regularization metrics for VAE."""
            preds, labels = eval_preds
            
            # Compute reconstruction metrics (same as AE)
            mse_loss = np.mean((preds - labels) ** 2)
            mae_loss = np.mean(np.abs(preds - labels))
            
            ss_res = np.sum((labels - preds) ** 2)
            ss_tot = np.sum((labels - labels.mean()) ** 2)
            r2_score = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Note: KL divergence would need to be computed during forward pass
            # and stored separately as it's not available from predictions alone
            
            return {
                'reconstruction_mse': mse_loss,
                'reconstruction_mae': mae_loss,
                'reconstruction_r2': r2_score,
                'latent_dimensionality': preds.shape[-1] if len(preds.shape) > 1 else 1,
            }

        return preprocess_logits_for_metrics, vae_metrics


class EmbeddingGeneratorMetricsHF(MetricsHF):
    """Metrics for embedding generator training."""
    
    def get_metrics(self) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]] | tuple[None, None]:
        def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
            """Convert generator outputs to predictions."""
            return logits.detach().cpu().numpy()

        def embedding_generator_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
            """Compute embedding prediction metrics."""
            preds, labels = eval_preds
            
            # Compute embedding prediction accuracy (MSE and cosine similarity)
            mse_loss = np.mean((preds - labels) ** 2)
            mae_loss = np.mean(np.abs(preds - labels))
            
            # Compute cosine similarity between predicted and target embeddings
            def cosine_similarity(a, b):
                norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
                norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
                return np.mean(np.sum((a / (norm_a + 1e-8)) * (b / (norm_b + 1e-8)), axis=-1))
            
            cosine_sim = cosine_similarity(preds, labels)
            
            # Compute embedding norm statistics
            pred_norm_mean = np.mean(np.linalg.norm(preds, axis=-1))
            pred_norm_std = np.std(np.linalg.norm(preds, axis=-1))
            target_norm_mean = np.mean(np.linalg.norm(labels, axis=-1))
            
            return {
                'embedding_mse': mse_loss,
                'embedding_mae': mae_loss,
                'embedding_cosine_similarity': cosine_sim,
                'pred_embedding_norm_mean': pred_norm_mean,
                'pred_embedding_norm_std': pred_norm_std,
                'target_embedding_norm_mean': target_norm_mean,
            }

        return preprocess_logits_for_metrics, embedding_generator_metrics


class EmbeddingCLLPMetricsHF(MetricsHF):
    """Metrics for embedding-conditioned CLLP training."""
    
    def get_metrics(self) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[EvalPrediction], dict]] | tuple[None, None]:
        def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
            """Convert action logits to predictions."""
            preds = torch.argmax(logits, dim=-1)
            return preds.detach().cpu().numpy()

        def embedding_cllp_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
            """Compute action prediction metrics for embedding CLLP."""
            preds, labels = eval_preds
            assert preds.shape == labels.shape
            
            # Compute action accuracy
            accuracy = (preds == labels).astype(float).mean().item()
            
            # Compute per-class accuracy if multiple actions
            num_classes = max(np.max(preds), np.max(labels)) + 1
            per_class_acc = {}
            
            for class_id in range(num_classes):
                mask = labels == class_id
                if np.sum(mask) > 0:
                    class_acc = (preds[mask] == labels[mask]).astype(float).mean().item()
                    per_class_acc[f'action_{class_id}_accuracy'] = class_acc
            
            # Compute top-k accuracy for k=3 if applicable
            if num_classes >= 3:
                # This would need logits, not just predictions, so approximate
                # by checking if prediction is within top candidates
                pass  # Skip for now as we only have argmax predictions
            
            metrics = {
                'action_accuracy': accuracy,
                'num_action_classes': num_classes,
            }
            metrics.update(per_class_acc)
            
            return metrics

        return preprocess_logits_for_metrics, embedding_cllp_metrics
