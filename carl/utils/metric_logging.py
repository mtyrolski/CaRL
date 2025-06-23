import numpy as np


class MetricsAccumulator:
    def __init__(self):
        # TODO: add type hints
        self._metrics: dict = {}
        self._data_to_average = {}
        self._data_to_sum = {}
        self._data_to_accumulate = {}

    def log_metric_to_average(self, name: str, value: float | int) -> None:
        self._data_to_average.setdefault(name, []).append(value)
        self._metrics[name] = np.mean(self._data_to_average[name])

    def log_metric_to_sum(self, name: str, value: float | int) -> None:
        self._data_to_sum.setdefault(name, []).append(value)
        self._metrics[name] = np.sum(self._data_to_sum[name])

    def return_scalars(self) -> dict[str, float | int]:
        return self._metrics

    def get_value(self, name: str) -> float | int:
        return self._metrics[name]
