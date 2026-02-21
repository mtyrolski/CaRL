from pathlib import Path

import joblib
import numpy as np
import torch

from carl.algorithms.solve_instances import SolveInstances
from carl.planners.base import Experience
from carl.planners.base import SearchInfo
from carl.planners.base import Solution


def _make_unsolved_experience() -> Experience:
    return Experience(
        solution=Solution(solved=False),
        search_info=SearchInfo(finished_reason="dummy"),
    )


class DummySolver:
    def __init__(self) -> None:
        self.construct_calls = 0
        self.solved: list[object] = []

    def construct_networks(self) -> None:
        self.construct_calls += 1

    def solve(self, problem: object) -> Experience:
        self.solved.append(problem)
        return _make_unsolved_experience()


class DummyLoader:
    def __init__(self, batches: list[object]) -> None:
        self._batches = batches

    def reset_dataloader(self):
        return iter(self._batches)


class DummyResultLogger:
    def __init__(self) -> None:
        self.logged: list[list[Experience]] = []

    def log_results(self, results: list[Experience]) -> None:
        self.logged.append(results)


def test_normalize_problems_handles_tensor_and_numpy():
    solver = DummySolver()
    loader = DummyLoader([])
    logger = DummyResultLogger()
    algo = SolveInstances(
        solver=solver,  # type: ignore[arg-type]
        data_loader=loader,  # type: ignore[arg-type]
        result_logger=logger,  # type: ignore[arg-type]
        problems_to_solve=0,
        n_parallel_workers=1,
    )

    tensor_batch = torch.tensor([[1, 2], [3, 4]])
    tensor_result = algo._normalize_problems(tensor_batch)
    assert len(tensor_result) == 2
    assert np.array_equal(tensor_result[0], np.array([1, 2]))

    numpy_batch = np.array([[5, 6], [7, 8]])
    numpy_result = algo._normalize_problems(numpy_batch)
    assert len(numpy_result) == 2
    assert np.array_equal(numpy_result[1], np.array([7, 8]))

    list_result = algo._normalize_problems([9, 10])
    assert list_result == [9, 10]


def test_run_dumps_each_batch_and_logs_results(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    solver = DummySolver()
    logger = DummyResultLogger()
    loader = DummyLoader([[1, 2], [3]])
    algo = SolveInstances(
        solver=solver,  # type: ignore[arg-type]
        data_loader=loader,  # type: ignore[arg-type]
        result_logger=logger,  # type: ignore[arg-type]
        problems_to_solve=10,
        n_parallel_workers=1,
        dump_solved=True,
        tag="unit",
    )

    algo.run()

    assert solver.construct_calls == 1
    assert solver.solved == [1, 2, 3]
    assert algo.completed_problems == 3
    assert len(logger.logged) == 2
    assert len(logger.logged[0]) == 2
    assert len(logger.logged[1]) == 1

    first_dump = tmp_path / "solve_attempts" / "solved_problems_unit_batch_1.joblib"
    second_dump = tmp_path / "solve_attempts" / "solved_problems_unit_batch_2.joblib"
    assert first_dump.exists()
    assert second_dump.exists()
    assert len(joblib.load(first_dump)) == 2
    assert len(joblib.load(second_dump)) == 1


def test_run_skips_existing_batch_dump(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dump_dir = tmp_path / "solve_attempts"
    dump_dir.mkdir()
    joblib.dump([_make_unsolved_experience()], dump_dir / "solved_problems_skip_batch_1.joblib")

    solver = DummySolver()
    logger = DummyResultLogger()
    loader = DummyLoader([[10, 11], [12]])
    algo = SolveInstances(
        solver=solver,  # type: ignore[arg-type]
        data_loader=loader,  # type: ignore[arg-type]
        result_logger=logger,  # type: ignore[arg-type]
        problems_to_solve=10,
        n_parallel_workers=1,
        dump_solved=False,
        tag="skip",
    )

    algo.run()

    assert solver.solved == [12]
    assert algo.completed_problems == 3
    assert len(logger.logged) == 1
    assert len(logger.logged[0]) == 1


def test_run_stops_immediately_when_budget_already_reached(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    solver = DummySolver()
    logger = DummyResultLogger()
    loader = DummyLoader([[1, 2]])
    algo = SolveInstances(
        solver=solver,  # type: ignore[arg-type]
        data_loader=loader,  # type: ignore[arg-type]
        result_logger=logger,  # type: ignore[arg-type]
        problems_to_solve=0,
        n_parallel_workers=1,
    )

    algo.run()

    assert solver.solved == []
    assert algo.completed_problems == 0
    assert logger.logged == []
