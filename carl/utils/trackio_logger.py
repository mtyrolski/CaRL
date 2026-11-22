"""Local Trackio logging without modifying the experiment's dependencies."""
import atexit
import json
import os
import re
import select
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers.trainer_callback import TrainerCallback

from carl.utils.logging_base import CaRLLogger

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = ROOT / 'local/experiments/trackio/.venv/bin/python'
DEFAULT_DIR = ROOT / 'analysis/trackio'


def _json_default(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


class MetricChannel:
    """A named metric stream."""

    def __init__(self, run: 'TrackioRun', key: str) -> None:
        self.run = run
        self.key = key

    def append(self, value: Any, step: int | None = None) -> None:
        self.run.log({self.key: value}, step=step)

    def log(self, value: Any, step: int | None = None) -> None:
        self.append(value, step=step)


class TrackioRun:
    """Send metrics to a local Trackio worker and flush them on close."""

    def __init__(self, project: str, name: str, config: Mapping[str, Any], python: str | Path | None = None) -> None:
        executable = Path(python or os.environ.get('CARL_TRACKIO_PYTHON', str(DEFAULT_PYTHON)))
        if not executable.is_file():
            raise RuntimeError(f'Trackio environment not found: {executable}. See README.md for setup.')
        env = {key: value for key, value in os.environ.items() if not key.startswith('TRACKIO_')}
        env.update({
            'TRACKIO_DIR': str(Path(os.environ.get('TRACKIO_DIR', str(DEFAULT_DIR))).resolve()),
            'HF_HUB_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1',
            'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
            'GRADIO_ANALYTICS_ENABLED': 'False',
        })
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT), env.get('PYTHONPATH', ''))))
        self._process = subprocess.Popen(
            [str(executable), '-m', 'carl.utils.trackio_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=ROOT, env=env,
        )
        self._closed = False
        self._write({'project': project, 'name': name, 'config': dict(config)})
        assert self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], 30)
        if not ready or self._process.stdout.readline().strip() != 'ready':
            self._process.terminate()
            self._process.wait(timeout=5)
            raise RuntimeError('Local Trackio worker failed to start')
        atexit.register(self.finish)

    def _write(self, event: Mapping[str, Any]) -> None:
        if self._closed or self._process.stdin is None:
            raise RuntimeError('Trackio run is closed')
        self._process.stdin.write(json.dumps(event, default=_json_default, allow_nan=False) + '\n')
        self._process.stdin.flush()

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        self._write({'type': 'log', 'metrics': dict(metrics), 'step': step})

    def __getitem__(self, key: str) -> MetricChannel:
        return MetricChannel(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.log({key: value})

    def finish(self) -> None:
        if self._closed:
            return
        try:
            self._write({'type': 'finish'})
        except (BrokenPipeError, OSError):
            pass
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            code = self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            code = self._process.wait(timeout=5)
        if code:
            raise RuntimeError(f'Local Trackio worker exited with code {code}')


class TrackioCallback(TrainerCallback):
    """Forward Hugging Face Trainer metrics to Trackio."""

    def __init__(self, run: TrackioRun, log_parameters: bool = False) -> None:
        self.run = run
        self.log_parameters = log_parameters

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.log_parameters and hasattr(args, 'to_sanitized_dict'):
            self.run['trainer/parameters'] = args.to_sanitized_dict()

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
        if logs:
            self.run.log(logs, step=state.global_step)

    def finish(self) -> None:
        self.run.finish()


class TrackioCaRLLogger(CaRLLogger):
    """Create a local Trackio run for CaRL trainers and solvers."""

    def __init__(self, name: str, description: str, project: str, tags: Any, log_parameters: bool = False) -> None:
        project_name = re.sub(r'[^A-Za-z0-9_-]+', '-', project).strip('-') or 'carl'
        tag_list = [tags] if isinstance(tags, str) else list(tags)
        self.run = TrackioRun(project_name, name, {'description': description, 'tags': tag_list, 'log_parameters': log_parameters})
        self._callback = TrackioCallback(self.run, log_parameters=log_parameters)

    def return_logger(self) -> TrackioCallback:
        return self._callback

    def finish(self) -> None:
        self.run.finish()
