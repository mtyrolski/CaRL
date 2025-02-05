from functools import reduce
from operator import or_
from typing import Any

Metrics = dict[str, float | list[float]]
from carl.solver.planners import Experience


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
