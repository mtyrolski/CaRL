"""Module for solving instances using a Solver and data loader, with optional parallelism and result logging."""
import os
from pickle import HIGHEST_PROTOCOL
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from joblib import Parallel, delayed, dump
from loguru import logger

from carl.environment.instance_generator import BasicInstanceGenerator
from carl.solver.subgoal_search import Solver
from carl.utils.result_loggers import ResultLogger
from carl.algorithms.algorithm import Algorithm




Problem = Union[str, np.ndarray]
Result = Tuple[Dict[str, Any], Dict[str, Any]]
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
    ) -> None:
        super().__init__()
        self.solver = solver
        self.data_loader = data_loader
        self.result_logger = result_logger
        self.problems_to_solve = problems_to_solve
        self.completed_problems: int = 0
        self.n_parallel_workers = n_parallel_workers
        self.dump_solved = dump_solved

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
            return np_array.tolist()
        if isinstance(problems, np.ndarray):  # raw array
            return problems.tolist()
        # fallback for list-like iterables
        return list(problems)

    def run(self) -> None:
        """
        Main execution loop: fetch batches, solve problems,
        log results, and optionally dump all experiences.
        """
        logger.warning("Starting SolveInstances.run()")
        self.solver.construct_networks()

        all_experiences: List[List[Result]] = []

        for batch_idx, problems in enumerate(self.data_loader.reset_dataloader()):
            # stop if we have reached the target count
            if self.completed_problems >= self.problems_to_solve:
                logger.info(f"Completed {self.completed_problems}/{self.problems_to_solve} problems. Stopping.")
                break

            # convert loader batch to a flat list of problems
            conv_problems = self._normalize_problems(problems)

            num_problems = len(conv_problems)
            logger.info(f"Batch {batch_idx + 1}: {num_problems} problems")

            results: List[Result]
            if self.n_parallel_workers == 1:
                results = []
                # sequential solve with progress logging
                for idx, problem in enumerate(conv_problems, start=1):
                    logger.info(
                        f"Solving problem {idx}/{num_problems} "
                        f"of batch {batch_idx + 1}"
                    )
                    result = self.solver.solve(problem)
                    results.append(result)
                    self.completed_problems += 1
            else:
                # parallel execution across workers
                results = Parallel(
                    n_jobs=self.n_parallel_workers,
                    verbose=50
                )(delayed(self.solver.solve)(prob) for prob in conv_problems)
                self.completed_problems += len(results)

            self.result_logger.log_results(results)
            all_experiences.append(results)
            logger.info(
                f"Total completed: {self.completed_problems}/{self.problems_to_solve}"
            )

        if self.dump_solved:
            dump(
                all_experiences,
                "solved_problems.joblib",
                protocol=HIGHEST_PROTOCOL,
            )
            logger.info("Dumped solved problems to 'solved_problems.joblib'")