"""Value prompts and discounted-return predictions."""
from collections.abc import Sequence

import numpy as np

from carl.icl.api import RequestFailed
from carl.icl.boards import from_rows
from carl.icl.llm_components.base import LLMComponentBase
from carl.icl.llm_components.formatting import board_distance, schema_object, value_question
from carl.icl.types import Board, JSONObject, TrainingContext, ValueExample


class LLMValueFunction(LLMComponentBase):
    """Build and validate task-value estimates."""

    def value_payload(self, state: Board) -> JSONObject:
        """Build a value request with discounted continuation-return examples."""
        contexts = self.training["contexts"][: self.config.few_shot]
        sources: Sequence[TrainingContext | ValueExample]
        if self.config.value_examples == "trajectory":
            if "value_examples" not in self.training:
                raise ValueError("Verified trajectory value examples have not been prepared")
            allowed = {context["id"] for context in contexts}
            trajectory_sources = [example for example in self.training["value_examples"] if example["context"] in allowed]
            if {example["context"] for example in trajectory_sources} != allowed:
                raise ValueError("Missing value examples for one or more training contexts")
            sources = trajectory_sources
        else:
            sources = contexts
        remaining_target = self.config.value_target == "remaining_actions"
        examples = []
        for source in sources:
            remaining = source["remaining_actions"]
            answer: JSONObject = {"remaining_actions": remaining} if remaining_target else {
                "value": self.config.discount ** remaining
            }
            examples.append((value_question(from_rows(source["state"]), features=self.config.value_board_features), answer))
        if remaining_target:
            prompt = self.config.prompts.value_remaining
            schema = schema_object({"remaining_actions": {"type": "integer", "minimum": 0, "maximum": 10000}})
        else:
            prompt = self.config.prompts.value + f"\nDiscount: {self.config.discount}."
            schema = schema_object({"value": {"type": "number", "minimum": 0, "maximum": 1}})
        if self.config.value_board_features:
            prompt += " Board features are derived from the board. task_solved is exact; box_to_goal_manhattan ignores walls."
        return self._payload(
            "value",
            prompt,
            examples,
            value_question(state, features=self.config.value_board_features),
            schema=schema,
        )
    def value(self, state: Board) -> float:
        """Return a finite discounted-return estimate in the unit interval."""
        if self.config.value_target == "remaining_actions":
            result = self._json(
                self.value_payload(state),
                lambda: {"remaining_actions": int(round(board_distance(state)))},
            )
            remaining = result.get("remaining_actions")
            if isinstance(remaining, bool) or not isinstance(remaining, int) or not 0 <= remaining <= 10000:
                raise RequestFailed("Value must return an integer remaining_actions in [0,10000]")
            return self.config.discount ** remaining
        result = self._json(
            self.value_payload(state),
            lambda: {"value": float(np.exp(-board_distance(state) / 16))},
        )
        value = result.get("value")
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise RequestFailed("Value must lie in [0,1]")
        return float(value)
