from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional, Tuple

import joblib
import numpy as np
from loguru import logger

from carl.algorithms.algorithm import Algorithm
from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer

NUM_ACTIONS = 4


def split_to_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a list into batches of size `batch_size`."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def find_optimal_solution(
    board: np.ndarray, max_depth: int = 5000, time_limit: int = 1_000_000
) -> Optional[List[int]]:
    """
    Depth-first search for the shortest Sokoban solution.
    Returns a list of actions or None if no solution within limits.
    """
    tokenizer = SokobanTokenizer()
    env = SokobanEnv(tokenizer)
    env.restore_full_state_from_np_array_version(board)

    stack: List[Tuple[np.ndarray, List[int]]] = [(env.get_state(), [])]
    best_solution: Optional[List[int]] = None
    best_length: int = max_depth + 1
    seen: set[Tuple[int, ...]] = {tuple(env.get_state().flatten())}

    iterations = 0
    while stack:
        state, actions = stack.pop()
        depth = len(actions)
        if depth >= best_length or depth > max_depth or iterations >= time_limit:
            break

        env.restore_full_state_from_np_array_version(state)
        for action in range(NUM_ACTIONS):
            new_state, _, done, _ = env.step(action)
            if np.array_equal(new_state, state):
                continue
            key = tuple(new_state.flatten())
            if key in seen:
                continue
            seen.add(key)

            if done:
                solution = actions + [action]
                if len(solution) < best_length:
                    best_solution = solution
                    best_length = len(solution)
                    logger.info(f"New best solution found: {best_length}")
            else:
                stack.append((new_state.copy(), actions + [action]))
            env.restore_full_state_from_np_array_version(state)

        iterations += 1

    return best_solution


def find_optimal_solution_breadth_first(
    board: np.ndarray, max_depth: int = 5000, time_limit: int = 1_000_000
) -> Optional[List[int]]:
    """
    Breadth-first search for the shortest Sokoban solution.
    Returns a list of actions or None if no solution within limits.
    """
    tokenizer = SokobanTokenizer()
    env = SokobanEnv(tokenizer)
    env.restore_full_state_from_np_array_version(board)

    queue: Deque[Tuple[np.ndarray, List[int]]] = deque([(env.get_state(), [])])
    seen: set[Tuple[int, ...]] = {tuple(env.get_state().flatten())}

    iterations = 0
    while queue and iterations < time_limit:
        state, actions = queue.popleft()
        depth = len(actions)
        if depth > max_depth:
            continue

        env.restore_full_state_from_np_array_version(state)
        for action in range(NUM_ACTIONS):
            new_state, _, done, _ = env.step(action)
            if np.array_equal(new_state, state):
                continue
            key = tuple(new_state.flatten())
            if key in seen:
                continue
            seen.add(key)

            next_actions = actions + [action]
            if done:
                logger.info(f"Solution found of length {len(next_actions)}")
                return next_actions

            queue.append((new_state.copy(), next_actions))
            env.restore_full_state_from_np_array_version(state)

        iterations += 1

    return None


class FindOptimalSolution(Algorithm):
    def __init__(
        self,
        path_to_list_of_board_states: str,
        n_jobs: int = 8,
        max_depth: int = 1_000,
        time_limit: int = 5_000_000,
        solver_idx: int = 0,
        solvers_count: int = 1,
    ) -> None:
        super().__init__()
        self.path_to_list_of_board_states = path_to_list_of_board_states
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.solver_idx = solver_idx
        self.solvers_count = solvers_count

    def run(self) -> None:
        """Load boards, split into batches, solve in parallel, and dump results."""
        all_boards: List[np.ndarray] = joblib.load(self.path_to_list_of_board_states)[:1000]
        batch_size = max(1, len(all_boards) // self.solvers_count)
        batches = split_to_batches(all_boards, batch_size)
        logger.info(f"Split boards into {len(batches)} batches")

        boards = batches[self.solver_idx]
        logger.info(f"Solver {self.solver_idx} will solve {len(boards)} boards")

        solutions = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(find_optimal_solution_breadth_first)(board, self.max_depth, self.time_limit)
            for board in boards
        )

        solved = [(idx, sol) for idx, sol in enumerate(solutions) if sol is not None]
        logger.info(f"Solved {len(solved)} out of {len(boards)} boards")

        combined = [(boards[idx], sol, idx) for idx, sol in solved]
        joblib.dump(combined, "xy_solutions.joblib")
        logger.info("Solutions saved to xy_solutions.joblib")
