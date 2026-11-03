"""Goal-conditioned policy prompts and action distributions."""
import json

import numpy as np

from carl.icl.api import RequestFailed
from carl.icl.boards import to_rows
from carl.icl.luna_components.base import LunaBase
from carl.icl.luna_components.formatting import (
    POLICY_COORDINATE_INSTRUCTIONS, POLICY_HORIZONS, board_coordinates, board_distance, schema_object,
)
from carl.icl.types import Board, Candidate, JSONObject, Probabilities


class LunaPolicy(LunaBase):
    """Build and validate the frozen goal-conditioned policy."""

    def policy_examples(self) -> list[tuple[JSONObject, JSONObject]]:
        """Pair verified next actions with varied future target horizons."""
        examples = []
        for index, context in enumerate(
            self.training["contexts"][: self.config.few_shot]
        ):
            horizon = POLICY_HORIZONS[index % len(POLICY_HORIZONS)]

            def distance(candidate: Candidate, offset: int = horizon) -> int:
                return abs(candidate["k"] - offset)

            example = min(
                context["candidates"],
                key=distance,
            )
            probs = [0.0] * 4
            probs[example["actions"][0]] = 1.0
            examples.append(
                (
                    {"state": context["state"], "subgoal": example["subgoal"]},
                    {"probabilities": probs},
                )
            )
        if len(examples) != self.config.few_shot:
            raise ValueError("Insufficient distinct policy few-shot contexts")
        return examples
    def policy_payload(self, state: Board, subgoal: Board) -> JSONObject:
        """Build the same four-action probability request for calibration and search."""
        schema = schema_object(
            {
                "probabilities": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                }
            }
        )
        payload = self._payload(
            "policy",
            self.config.prompts.policy,
            self.policy_examples(),
            {"state": to_rows(state), "subgoal": to_rows(subgoal)},
            schema=schema,
        )
        if self.config.policy_coordinate_features:
            payload["input"][0]["content"][0]["text"] += POLICY_COORDINATE_INSTRUCTIONS
            for message in payload["input"]:
                if message["role"] == "user":
                    content = message["content"][0]
                    question = json.loads(content["text"])
                    for key in ("state", "subgoal"):
                        question[f"{key}_coordinates"] = board_coordinates(question[key])
                    content["text"] = json.dumps(question)
        return payload
    def probabilities(self, state: Board, subgoal: Board) -> Probabilities:
        """Validate and temperature-normalize the sampled policy distribution."""

        def mock() -> JSONObject:
            scores = np.array(
                [
                    -board_distance(np.asarray(self.env.next_state(state, a)), subgoal)
                    for a in range(4)
                ]
            )
            scores -= scores.max()
            probabilities = np.exp(scores * 4)
            return {"probabilities": (probabilities / probabilities.sum()).tolist()}

        result = self._json(self.policy_payload(state, subgoal), mock)
        try:
            raw = result["probabilities"]
            if not isinstance(raw, list) or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in raw
            ):
                raise ValueError("Policy probabilities must be JSON numbers")
            probabilities = np.asarray(raw, dtype=float)
            if (
                probabilities.shape != (4,)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or probabilities.sum() <= 0
            ):
                raise ValueError("Invalid probabilities")
            probabilities /= probabilities.max()
            probabilities = probabilities ** (1 / self.config.policy_temperature)
            normalized: Probabilities = probabilities / probabilities.sum()
            return normalized
        except (KeyError, TypeError, ValueError):
            raise RequestFailed(
                "Policy must return four finite nonnegative probabilities with positive sum"
            ) from None
