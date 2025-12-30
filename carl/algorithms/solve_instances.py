"""Module for solving instances using a Solver and data loader, with optional parallelism and result logging."""
import os
from pickle import HIGHEST_PROTOCOL
from typing import Any, List, Union

import numpy as np
import torch
from joblib import Parallel, delayed, dump
from loguru import logger

from carl.environment.instance_generator import BasicInstanceGenerator
from carl.solver.subgoal_search import Solver
from carl.utils.result_loggers import ResultLogger
from carl.algorithms.algorithm import Algorithm
from typing_extensions import TypeAlias
from carl.utils.aliases import State
from carl.planners.base import Experience
from os.path import join
Problem: TypeAlias = State
Result: TypeAlias = Experience
CUDA_VISIBLE_DEVICES__ENV_VAR = 'CUDA_VISIBLE_DEVICES'


class SolveInstances(Algorithm):
    """
    Algorithm that retrieves problem instances from a data loader
    and solves them using the provided Solver, either sequentially
    or in parallel, then logs and optionally dumps results.
    """

    def __init__(
        self,
        solver: Solver,
        data_loader: BasicInstanceGenerator,
        result_logger: ResultLogger,
        problems_to_solve: int,
        n_parallel_workers: int,
        dump_solved: bool = False,
        tag: str | None = None,
    ) -> None:
        super().__init__()
        self.solver = solver
        self.data_loader = data_loader
        self.result_logger = result_logger
        self.problems_to_solve = problems_to_solve
        self.completed_problems: int = 0
        self.n_parallel_workers = n_parallel_workers
        self.dump_solved = dump_solved
        self.tag = tag

        cuda_devices = os.environ.get(CUDA_VISIBLE_DEVICES__ENV_VAR, '')
        if cuda_devices and self.n_parallel_workers > 1:
            logger.warning(
                f"{CUDA_VISIBLE_DEVICES__ENV_VAR} is set to '{cuda_devices}' "
                f"but parallel workers > 1 ({self.n_parallel_workers})."
            )
            logger.info("Proceeding with configured parallelism.")
        else:
            logger.info(f"Using {self.n_parallel_workers} parallel worker(s)")

    def _normalize_problems(
        self, problems: Union[torch.Tensor, np.ndarray, Any]
    ) -> List[Problem]:
        """
        Convert incoming batch of problems into a list of Problem items.
        Handles torch.Tensor, numpy.ndarray, or other iterable types.
        """
        if isinstance(problems, torch.Tensor):  # from GPU
            np_array = problems.detach().cpu().numpy()
            return list(np_array)
        if isinstance(problems, np.ndarray):  # raw array
            return list(problems)
        # fallback for list-like iterables
        return list(problems)

    def run(self) -> None:
        """
        Main execution loop: fetch batches, solve problems,
        log results, and optionally dump all experiences.
        """
        logger.warning("Starting SolveInstances.run()")
        self.solver.construct_networks()

        all_experiences: List[List[Experience]] = []

        for batch_idx, problems in enumerate(self.data_loader.reset_dataloader()):
            # stop if we have reached the target count
            if self.completed_problems >= self.problems_to_solve:
                logger.info(f"Completed {self.completed_problems}/{self.problems_to_solve} problems. Stopping.")
                break

            # convert loader batch to a flat list of problems
            conv_problems: list[Problem] = self._normalize_problems(problems)

            num_problems = len(conv_problems)
            logger.info(f"Batch {batch_idx + 1}: {num_problems} problems")

            results: List[Experience]
            if self.n_parallel_workers == 1:
                results = []
                # sequential solve with progress logging
                for idx, problem in enumerate(conv_problems, start=1):
                    logger.info(
                        f"Solving problem {idx}/{num_problems} (of type {type(problem).__name__})"
                        f"of batch {batch_idx + 1}"
                    )
                    result = self.solver.solve(problem)
                    results.append(result)
                    self.completed_problems += 1
            else:
                # parallel execution across workers
                results = Parallel( # type: ignore
                    n_jobs=self.n_parallel_workers,
                    verbose=50
                )(delayed(self.solver.solve)(prob) for prob in conv_problems)
                self.completed_problems += len(results)

            self.result_logger.log_results(results)
            all_experiences.append(results)
            logger.info(
                f"Total completed: {self.completed_problems}/{self.problems_to_solve}"
            )
            folder_name = 'solved_problems'
            os.makedirs(folder_name, exist_ok=True)
            if self.dump_solved:
                batch_number = batch_idx + 1
                if self.tag:
                    filename = f"solved_problems_{self.tag}_batch_{batch_number}.joblib"
                else:
                    filename = f"solved_problems_batch_{batch_number}.joblib"

                dump(
                    results,
                    join(folder_name, filename),
                    protocol=HIGHEST_PROTOCOL,
                )
                logger.info(f"Dumped solved problems for batch {batch_number} to '{join(folder_name, filename)}'")