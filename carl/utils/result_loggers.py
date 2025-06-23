from abc import ABC
from abc import abstractmethod
from os.path import join
from typing import Any
import os
from loguru import logger
import numpy as np
from carl.planners.base import Experience
from carl.utils.loggers import NeptuneCaRLLogger
from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.utils.metric_logging import MetricsAccumulator


class ResultLogger(ABC):
    @abstractmethod
    def log_results(self, results: Any) -> None:
        raise NotImplementedError

    @property
    def node_global_id(self) -> str:
        """
        Support for heterogeneous jobs.
        """
        het_group_id = os.environ.get('CARL_HET_GROUP_ID', None)
        local_worker_id = os.environ.get('CARL_LOCAL_WORKER_ID', None)

        if het_group_id is None or local_worker_id is None:
            logger.debug('No HET_GROUP_ID or LOCAL_WORKER_ID found in the environment.')
            return ''

        return f'HG{het_group_id}_W{local_worker_id}'


class SubgoalSearchResultLogger(ResultLogger):
    def __init__(self, custom_logger: NeptuneCaRLLogger, budget_logs: list[int], problem_to_solve: int) -> None:
        self.custom_logger = custom_logger.return_logger()
        if not hasattr(self.custom_logger, 'run') or self.custom_logger.run is None or not hasattr(self.custom_logger.run, '__getitem__'):
            raise RuntimeError("Custom logger is not properly initialized: 'run' attribute is missing or not subscriptable.")
        self.completed_problems: int = 0
        self.solved_problems: int = 0
        self.problem_to_solve = problem_to_solve
        self.finished_reasons: dict[str, int] = {}
        self.budget_logs = budget_logs
        self.total_completed_problems = 0
        self.solved_stats = MetricsAccumulator()

    def log_results(self, results: list[Experience]) -> None:
        base_completed_problems = self.completed_problems
        self.completed_problems = self.completed_problems + len(results)
        assert self.custom_logger.run is not None, "Custom logger is not initialized."
        self.custom_logger.run[join('total_completed_problems', self.node_global_id)].append(
            step=self.completed_problems, value=self.completed_problems)
        SokobanEnv(tokenizer=SokobanTokenizer())

        for task_id, experience in enumerate(results):
            solution = experience.solution
            search_info = experience.search_info
            self._log_solved_rate(solution)
            self.solved_problems += int(solution.solved)
            solution_length, path_length_all_nodes = self._get_solution_lengths(solution)
            if solution.solved:
                self._log_solved_instance_metrics(base_completed_problems, task_id, solution_length, path_length_all_nodes, search_info)
            self._log_general_metrics(base_completed_problems, task_id, solution_length, path_length_all_nodes, search_info)
            self._log_budget_solved_rates(solution, search_info)
            self._log_additional_search_metrics(base_completed_problems, task_id, search_info)
            self._count_finished_reason(search_info)
        self._log_final_solved_rates()
        self._log_final_finished_reasons()
        if self.completed_problems == self.problem_to_solve:
            self._log_success_rates_by_budget()

    def _run(self):
        if not hasattr(self.custom_logger, 'run') or self.custom_logger.run is None or not hasattr(self.custom_logger.run, '__getitem__'):
            raise RuntimeError("Custom logger is not properly initialized: 'run' attribute is missing or not subscriptable.")
        return self.custom_logger.run

    def _log_solved_rate(self, solution):
        self.solved_stats.log_metric_to_average('rate/full', int(solution.solved))

    def _get_solution_lengths(self, solution):
        solution_length = 0
        path_length_all_nodes = 0
        # You may want to implement actual logic for these lengths if needed
        return solution_length, path_length_all_nodes

    def _log_solved_instance_metrics(self, base_completed_problems, task_id, solution_length, path_length_all_nodes, search_info):
        self._run()['solved_instances_solution_path_length_all_nodes'].append(
            step=base_completed_problems + task_id, value=path_length_all_nodes + 1)
        self._run()['solved_instances_solution_path_length'].append(
            step=base_completed_problems + task_id, value=solution_length)
        self._run()['solved_instances_tree_size'].append(
            step=base_completed_problems + task_id, value=search_info.tree_size)
        self._run()['solved_instances_tree_depth'].append(
            step=base_completed_problems + task_id, value=search_info.tree_depth)
        self._run()['solved_instances_leaf_nodes'].append(
            step=base_completed_problems + task_id, value=search_info.leaf_nodes)
        self._run()['solved_instances_branching_factor'].append(
            step=base_completed_problems + task_id, value=search_info.branching_factor)
        self.solved_stats.log_metric_to_average("average_solved_instances_solution_path_length", solution_length)
        self.solved_stats.log_metric_to_average("average_solved_instances_tree_size", search_info.tree_size)
        self.solved_stats.log_metric_to_average("average_solved_instances_tree_depth", search_info.tree_depth)
        self.solved_stats.log_metric_to_average("average_solved_instances_leaf_nodes", search_info.leaf_nodes)
        self.solved_stats.log_metric_to_average("average_solved_instances_branching_factor", search_info.branching_factor)
        self.solved_stats.log_metric_to_average("average_solved_instances_solution_path_length_all_nodes", path_length_all_nodes)

    def _log_general_metrics(self, base_completed_problems, task_id, solution_length, path_length_all_nodes, search_info):
        self._run()['solution_path_length'].append(
            step=base_completed_problems + task_id, value=solution_length)
        self.solved_stats.log_metric_to_average("average_solution_path_length", solution_length)
        self.solved_stats.log_metric_to_average("average_tree_size", search_info.tree_size)
        self.solved_stats.log_metric_to_average("average_tree_depth", search_info.tree_depth)
        self.solved_stats.log_metric_to_average("average_leaf_nodes", search_info.leaf_nodes)
        self.solved_stats.log_metric_to_average("average_branching_factor", search_info.branching_factor)
        self.solved_stats.log_metric_to_average("average_solved_solution_path_length_all_nodes", path_length_all_nodes)

    def _log_budget_solved_rates(self, solution, search_info):
        for budget in self.budget_logs:
            is_solved_in_budget = solution.solved and (search_info.low_level_nodes_visited <= budget)
            self.solved_stats.log_metric_to_average(f'rate/{budget}_nodes', is_solved_in_budget)
            # Optionally add high-level budget logic here

    def _log_additional_search_metrics(self, base_completed_problems, task_id, search_info):
        for metric, value in search_info.__dict__.items():
            if isinstance(value, (int, float)):
                self._run()[join(metric, self.node_global_id)].append(
                    step=base_completed_problems + task_id, value=value)

    def _count_finished_reason(self, search_info):
        finished_reason = search_info.finished_reason
        self.finished_reasons[finished_reason] = self.finished_reasons.get(finished_reason, 0) + 1

    def _log_final_solved_rates(self):
        self._run()[join('problems/solved', self.node_global_id)].append(
            step=self.completed_problems, value=self.solved_problems)
        self._run()[join('solved', self.node_global_id)].append(
            step=self.completed_problems, value=self.solved_stats.return_scalars())

    def _log_final_finished_reasons(self):
        for finished_reason, count in self.finished_reasons.items():
            self._run()[join(f'finished_reasons/{finished_reason}/rate', self.node_global_id)].append(
                step=self.completed_problems, value=count / self.completed_problems)

    def _log_success_rates_by_budget(self):
        for budget in self.budget_logs:
            self._run()['success_rate/budget'].append(
                step=budget, value=self.solved_stats.get_value(f'rate/{budget}_nodes'))
            self._run()['success_rate/budget_logscale'].append(
                step=int(np.log(budget) * 100), value=self.solved_stats.get_value(f'rate/{budget}_nodes'))
