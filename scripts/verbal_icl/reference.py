"""Keep task-value demonstrations separate from short verbal training snippets."""
from typing import Any

from carl.icl.boards import from_rows
from carl.icl.config import ExperimentConfig, load_config
from carl.icl.datasets import environment, load_data
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.components import ICLComponents
from carl.icl.types import TrainingData


def reference_data(spec: dict[str, Any], model: str) -> tuple[ExperimentConfig, TrainingData]:
    """Load the original frozen full-task reference contract unchanged."""
    config = load_config(spec.get('reference_config', spec['base_config']))
    if config.model != model:
        raise ValueError('Verbal and reference components must use the same Luna model')
    training, _instances = load_data(config)
    return config, training


def build_reference(config: ExperimentConfig, training: TrainingData, api: ExperimentAPI) -> ICLComponents:
    """Constructor verifies the frozen policy; V retains full solution continuation labels."""
    initial = from_rows(training['contexts'][0]['state'])
    return ICLComponents(config, api, environment(initial), training)
