"""Add verified task-progress demonstrations without assigning invented scale labels."""

import copy
import json
from collections.abc import Sequence
from typing import TypedDict

from carl.icl.boards import from_rows
from carl.icl.config import ExperimentConfig
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.luna_components.formatting import task
from carl.icl.responses import text_message
from carl.icl.types import Board, BoardEnvironment, BoardRows, JSONObject, TrainingData


class MechanicsExample(TypedDict):
    """A training transition illustrating task progress, without temporal calibration."""

    id: str
    state: BoardRows
    subgoal: BoardRows


def with_progress_examples(
    payload: JSONObject, examples: Sequence[MechanicsExample], *, layout_schema: bool, scale_label: str = "u"
) -> JSONObject:
    """Insert train-only complete-board pairs before the calibrated scale examples."""
    result = copy.deepcopy(payload)
    messages = [result["input"][0], text_message(
        "developer",
        "The next examples show useful task progress: pushing a box onto a goal. "
        "They are mechanics demonstrations without a measured temporal-scale label. "
        "Learn the objective and the symbol updates from them; do not infer u or d from them.",
    )]
    for example in examples:
        question = {"demonstration_kind": "task_progress", "state": example["state"],
                    "goal": task(from_rows(example["state"]))}
        target = ({f"row_{index:02d}": row for index, row in enumerate(example["subgoal"])}
                  if layout_schema else {"rows": example["subgoal"]})
        messages.extend((text_message("user", json.dumps(question)),
                         text_message("assistant", json.dumps({"subgoals": [target]}))))
    messages.append(text_message(
        "developer",
        "The following examples retain their measured scale labels. Use them to interpret "
        f"the requested {scale_label}. The final user message is the query to solve.",
    ))
    result["input"] = [*messages, *result["input"][1:]]
    return result


class TaskProgressComponents(LunaComponents):
    """Keep the production parser, rejection feedback, CLLP and value unchanged."""

    def __init__(
        self, config: ExperimentConfig, api: ExperimentAPI, env: BoardEnvironment,
        training: TrainingData, demonstrations: Sequence[MechanicsExample], *, scale_label: str = "u",
    ) -> None:
        super().__init__(config, api, env, training)
        self.progress_examples = tuple(demonstrations)
        self.scale_label = scale_label

    def generator_payload(
        self, state: Board, scale: float, resolve_embeddings: bool = True
    ) -> JSONObject:
        payload = super().generator_payload(state, scale, resolve_embeddings)
        return with_progress_examples(payload, self.progress_examples,
                                      layout_schema=self.config.generator_layout_schema, scale_label=self.scale_label)
