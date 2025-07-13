"""
Module for accumulating and retrieving training metrics, supporting mean and sum.
"""

import numpy as np
from typing import Dict, List, Union


class MetricsAccumulator:
    """
    Collect and aggregate numeric metrics over multiple steps.
    Supports logging metrics to average or sum, and retrieval of scalars.
    """

    _metrics: Dict[str, Union[int, float]]
    _data_to_average: Dict[str, List[Union[int, float]]]
    _data_to_sum: Dict[str, List[Union[int, float]]]
    _data_to_accumulate: Dict[str, List[Union[int, float]]]

    def __init__(self) -> None:
        """Initialize empty storage for metrics and raw data lists."""
        self._metrics = {}
        self._data_to_average = {}
        self._data_to_sum = {}
        self._data_to_accumulate = {}

    def log_metric_to_average(self, name: str, value: float | int) -> None:
        """Add a value and update the running average for the named metric."""
        self._data_to_average.setdefault(name, []).append(value)
        self._metrics[name] = np.mean(self._data_to_average[name]).item()

    def log_metric_to_sum(self, name: str, value: float | int) -> None:
        """Add a value and update the running sum for the named metric."""
        self._data_to_sum.setdefault(name, []).append(value)
        self._metrics[name] = np.sum(self._data_to_sum[name]).item()

    def return_scalars(self) -> dict[str, float | int]:
        """Return the current aggregated metric values."""
        return self._metrics

    def get_value(self, name: str) -> float | int:
        """Retrieve the aggregated value for a specific metric."""
        return self._metrics[name]
