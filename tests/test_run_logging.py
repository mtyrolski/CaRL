"""The experiment runner does not require remote tracking credentials."""
import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from carl import run as carl_run
from carl.algorithms.algorithm import Algorithm


def test_run_without_tracking_token(monkeypatch, tmp_path) -> None:
    executed = []

    class Grid:
        def __init__(self, config):
            pass

        def __len__(self):
            return 1

        def iter_grid_without_workers(self):
            yield {'algorithm': {}}

    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.endswith('_API_TOKEN'):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(carl_run, 'CarlGrid', Grid)
    monkeypatch.setattr(carl_run, '_instantiate_and_run', executed.append)
    carl_run.run(OmegaConf.create({'algorithm': {}}))
    assert len(executed) == 1


def test_logger_finishes_after_algorithm_failure(monkeypatch) -> None:
    closed = []

    class FailingAlgorithm(Algorithm):
        result_logger = SimpleNamespace(custom_logger=SimpleNamespace(finish=lambda: closed.append(True)))

        def run(self):
            raise RuntimeError('algorithm failed')

    monkeypatch.setattr(carl_run.hydra.utils, 'instantiate', lambda config: FailingAlgorithm())
    monkeypatch.setattr(carl_run, 'handle_logging', lambda *args: None)
    monkeypatch.setattr(carl_run, 'handle_precision', lambda *args: None)
    monkeypatch.setattr(carl_run.sys, 'getrecursionlimit', lambda: 2147483640)
    with pytest.raises(RuntimeError, match='algorithm failed'):
        carl_run._instantiate_and_run(OmegaConf.create({'algorithm': {}}))
    assert closed == [True]


def test_logger_finishes_after_setup_failure(monkeypatch) -> None:
    closed = []

    class AlgorithmWithLogger(Algorithm):
        result_logger = SimpleNamespace(custom_logger=SimpleNamespace(finish=lambda: closed.append(True)))

        def run(self):
            raise AssertionError('must not run')

    def fail_setup(*args):
        raise RuntimeError('setup failed')

    monkeypatch.setattr(carl_run.hydra.utils, 'instantiate', lambda config: AlgorithmWithLogger())
    monkeypatch.setattr(carl_run, 'handle_logging', fail_setup)
    with pytest.raises(RuntimeError, match='setup failed'):
        carl_run._instantiate_and_run(OmegaConf.create({'algorithm': {}}))
    assert closed == [True]
