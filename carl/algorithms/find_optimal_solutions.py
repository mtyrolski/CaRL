from typing import Any

import joblib
import numpy as np
from loguru import logger

from carl.algorithms.algorithm import Algorithm
from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer


def find_optimal_solution(board: np.ndarray, max_depth=5000, time_limit=1_000_000) -> list[int]:    # list of actions
    tokenizer = SokobanTokenizer()
    env = SokobanEnv(tokenizer)

    env.restore_full_state_from_np_array_version(board)

    stack = []
    stack.append((env.get_state(), []))

    best_solution = None
    best_solution_length = None
    seen_states = set()

    seen_states.add(tuple(env.get_state().flatten()))

    time = 0

    while len(stack) > 0:
        state, actions = stack.pop()

        if len(actions) > max_depth or (best_solution is not None and len(actions) >= len(best_solution)):
            continue

        if time > time_limit:
            break

        env.restore_full_state_from_np_array_version(state)

        for action in range(4):
            new_state, _, is_done, _ = env.step(action)

            if np.array_equal(new_state, state):
                env.restore_full_state_from_np_array_version(state)
                continue

            if tuple(new_state.flatten()) in seen_states:
                env.restore_full_state_from_np_array_version(state)
                continue

            seen_states.add(tuple(new_state.flatten()))

            if is_done:
                if best_solution is None or len(actions) < best_solution_length:
                    best_solution = actions + [action]
                    best_solution_length = len(best_solution)
                    logger.info(f'New best solution found: {best_solution_length}')
            else:
                stack.append((new_state.copy(), actions + [action]))

            # Restoring the old state
            env.restore_full_state_from_np_array_version(state)
        time += 1

    return best_solution


def find_optimal_solution_breath_first(board: np.ndarray,
                                       max_depth=5000,
                                       time_limit=1_000_000) -> list[int]:    # list of actions
    tokenizer = SokobanTokenizer()
    env = SokobanEnv(tokenizer)

    env.restore_full_state_from_np_array_version(board)

    queue = []
    queue.append((env.get_state(), []))
    seen_states = set()
    seen_states.add(tuple(env.get_state().flatten()))
    time = 0

    while len(queue) > 0:
        state, actions = queue.pop(0)

        if len(actions) > max_depth:
            continue

        if time > time_limit:
            break

        env.restore_full_state_from_np_array_version(state)

        for action in range(4):
            new_state, _, is_done, _ = env.step(action)

            if np.array_equal(new_state, state):
                env.restore_full_state_from_np_array_version(state)
                continue

            if tuple(new_state.flatten()) in seen_states:
                env.restore_full_state_from_np_array_version(state)
                continue

            seen_states.add(tuple(new_state.flatten()))

            if is_done:
                print('Solution found of len', len(actions) + 1)
                return actions + [action]
            else:
                queue.append((new_state.copy(), actions + [action]))

            # Restoring the old state
            env.restore_full_state_from_np_array_version(state)
        time += 1
    return None


def split_to_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


class FindOptimalSolution(Algorithm):
    def __init__(
        self,
        path_to_list_of_board_states: str,
        n_jobs: int = 8,
        max_depth: int = 1_000,
        time_limit: int = 5_000_000,
        solver_idx: int = 0,
        solvers_count: int = 1,
    ):
        super().__init__()
        self.path_to_list_of_board_states = path_to_list_of_board_states
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.solver_idx = solver_idx
        self.solvers_count = solvers_count

    def run(self) -> None:
        _loaded_boards = joblib.load(self.path_to_list_of_board_states)[:1000]    # pierwsze 1k nas interesuje

        batches = split_to_batches(_loaded_boards, len(_loaded_boards) // self.solvers_count)
        logger.info(f'Split boards into {len(batches)} batches')

        boards = batches[self.solver_idx]
        logger.info(f'Solver {self.solver_idx} will solve {len(boards)} boards')

        solutions = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(find_optimal_solution_breath_first)(board, self.max_depth, self.time_limit)
            for board in boards)

        solved_idxs = [idx for idx, solution in enumerate(solutions) if solution is not None]
        logger.info(f'Solved {len(solved_idxs)} out of {len(boards)} boards')

        solved_boards = [boards[idx] for idx in solved_idxs]
        solved_solutions = [solutions[idx] for idx in solved_idxs]

        combined_solutions = list(zip(solved_boards, solved_solutions, solved_idxs))

        joblib.dump(combined_solutions, 'xy_solutions.joblib')
        logger.info('Solutions saved to xy_solutions.joblib')
