"""Tests for local experiment tracking adapters."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

from carl.utils import trackio_logger


class FakePipe(io.StringIO):
    def close(self) -> None:
        self.saved = self.getvalue()
        super().close()


class FakeProcess:
    def __init__(self, *args, **kwargs) -> None:
        self.stdin = FakePipe()
        self.stdout = io.StringIO('ready\n')
        self.env = kwargs['env']
        self.command = args[0]

    def wait(self, timeout=None) -> int:
        return 0

    def terminate(self) -> None:
        pass


def test_local_worker_drops_remote_settings_and_logs(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / 'python'
    executable.touch()
    processes = []

    def spawn(*args, **kwargs):
        process = FakeProcess(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(trackio_logger.subprocess, 'Popen', spawn)
    monkeypatch.setattr(trackio_logger.select, 'select', lambda r, w, x, timeout: (r, [], []))
    monkeypatch.setenv('TRACKIO_SPACE_ID', 'user/remote')
    monkeypatch.setenv('TRACKIO_SERVER_URL', 'https://example.invalid')
    monkeypatch.setenv('TRACKIO_DIR', str(tmp_path / 'logs'))
    run = trackio_logger.TrackioRun('carl', 'test', {'source': 'test'}, python=executable)
    run['accuracy'].append(0.75, step=3)
    run['nested'].log({'success': 1}, step=4)
    run['parameters'] = {'seed': 5}
    run.finish()
    run.finish()

    process = processes[0]
    assert 'TRACKIO_SPACE_ID' not in process.env
    assert 'TRACKIO_SERVER_URL' not in process.env
    assert process.env['HF_HUB_OFFLINE'] == '1'
    assert process.env['TRACKIO_DIR'] == str(tmp_path / 'logs')
    events = [json.loads(line) for line in process.stdin.saved.splitlines()]
    assert events[1] == {'type': 'log', 'metrics': {'accuracy': 0.75}, 'step': 3}
    assert events[2] == {'type': 'log', 'metrics': {'nested': {'success': 1}}, 'step': 4}
    assert events[3] == {'type': 'log', 'metrics': {'parameters': {'seed': 5}}, 'step': None}
    assert events[4] == {'type': 'finish'}


def test_trainer_callback_forwards_metrics_and_parameters() -> None:
    class Run:
        def __init__(self) -> None:
            self.calls = []
            self.values = {}

        def __setitem__(self, key, value) -> None:
            self.values[key] = value

        def log(self, metrics, step=None) -> None:
            self.calls.append((metrics, step))

    run = Run()
    callback = trackio_logger.TrackioCallback(run, log_parameters=True)  # type: ignore[arg-type]
    callback.on_train_begin(SimpleNamespace(to_sanitized_dict=lambda: {'learning_rate': 0.1}), None, None)
    callback.on_log(None, SimpleNamespace(global_step=5), None, logs={'loss': 0.3})
    assert run.values == {'trainer/parameters': {'learning_rate': 0.1}}
    assert run.calls == [({'loss': 0.3}, 5)]


def test_logger_configurations_match_constructor() -> None:
    """Archived and active Hydra configs must not pass obsolete logger arguments."""
    import inspect

    import yaml

    allowed = set(inspect.signature(trackio_logger.TrackioCaRLLogger).parameters)
    configurations = []

    def collect(value):
        if isinstance(value, dict):
            if value.get('_target_') == 'carl.utils.loggers.TrackioCaRLLogger':
                configurations.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    root = Path(__file__).resolve().parents[2] / 'configs'
    for path in root.rglob('*.yaml'):
        text = path.read_text()
        if 'carl.utils.loggers.TrackioCaRLLogger' in text:
            collect(yaml.safe_load(text))
    assert configurations
    for config in configurations:
        assert set(config) - {'_target_'} <= allowed
