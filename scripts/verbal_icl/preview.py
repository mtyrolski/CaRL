"""Render exact controller and judge payloads without accessing an API."""
import json
from typing import Any, cast

from carl.icl.boards import from_rows
from carl.icl.config import ExperimentConfig
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.components import ICLComponents
from scripts.verbal_icl.components import VerbalComponents
from scripts.verbal_icl.demonstrations import VerbalDemo


class RecordingAPI:
    """Local payload recorder; never routes to the provider or client cache."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def call(self, _endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
        del mock
        self.payloads.append(payload)
        response = {'policy': {'action': 'UP'},
                    'judge': {'reached': False, 'reason': 'Preview only; no model evaluated this board.'}}
        return {'output_text': json.dumps(response[payload['metadata']['component']])}


def controller_payloads(config: ExperimentConfig, prompts: dict[str, str],
                        demos: list[VerbalDemo]) -> list[dict[str, Any]]:
    """Freeze exact worded train examples and one illustrative source/current query."""
    recorder = RecordingAPI()
    # The demonstration-backed policy/judge paths do not access the legacy reference.
    component = VerbalComponents(config, cast(ExperimentAPI, recorder), cast(ICLComponents, None),
                                  [], prompts, demonstrations=demos)
    row = demos[0]
    source, current = from_rows(row['source']), from_rows(row['state'])
    component.action(current, row['goal'], source)
    component.judge(current, row['goal'], source)
    return recorder.payloads
