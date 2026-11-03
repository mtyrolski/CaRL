"""Explicit-state subgoal selection, prompts, and proposal decoding."""
import json

import numpy as np

from carl.icl.api import RequestFailed
from carl.icl.boards import embedding_text, from_rows, to_rows, validate_subgoal
from carl.icl.luna_components.base import LunaBase
from carl.icl.luna_components.formatting import (
    FIELDS, board_distance, layout_generator_payload, schema_object, task,
)
from carl.icl.responses import text_message
from carl.icl.types import Board, BoardRows, Example, JSONObject, Probabilities


class LunaGenerator(LunaBase):
    """Generate and filter complete Sokoban subgoal boards."""

    def generator_examples(
        self, state: Board, scale: float, resolve_embeddings: bool
    ) -> list[Example]:
        """Select distinct contexts with matched weighted targets for relative/absolute labels."""
        if self.generator_checkpoint is not None:
            examples = self.generator_checkpoint.examples
            limit = self.config.generator_example_limit
            if limit is not None:
                if limit > len(examples):
                    raise ValueError(f"Generator checkpoint has {len(examples)} examples, fewer than requested {limit}")
                examples = examples[:limit]
            return [example.as_example() for example in examples]
        if self.config.few_shot == 0:
            return []
        method = self.config.method
        if method == "fixed":
            candidates = [
                example for example in self.raw_examples if example["k"] == int(scale)
            ]
        else:
            if self.calibrated is None or self.calibrated.get("status") != "completed":
                raise ValueError(
                    "Relative/absolute generation requires completed calibration"
                )
            if self.calibrated["policy_fingerprint"] != self.fingerprint():
                raise ValueError(
                    "Calibration does not match the frozen policy, examples or execution cap"
                )
            expected_provider = "mock" if self.api.provider == "mock" else "openai"
            if self.calibrated.get("provider") != expected_provider:
                raise ValueError(
                    "Mock calibration cannot be used for live inference, or vice versa"
                )
            candidates = list(self.calibrated["examples"])
            selected_ids = self.calibrated.get("selected_example_ids")
            if selected_ids is not None:
                # Calibration sampled one weighted target per context exactly once.
                # Every label condition reads the same saved IDs in the same order.
                by_id = {(row["context"], row.get("target_index")): row for row in candidates}
                ids = [(row["context"], row["target_index"]) for row in selected_ids]
                if len(ids) != len(set(ids)) or len({context for context, _ in ids}) != len(ids):
                    raise ValueError("Calibration selected duplicate demonstration IDs or contexts")
                if any(key not in by_id for key in ids):
                    raise ValueError("Calibration selected demonstration is missing from retained examples")
                candidates = [by_id[key] for key in ids]
                if self.config.selection == "direct":
                    if len(candidates) < self.config.few_shot:
                        raise ValueError(f"Need {self.config.few_shot} distinct few-shot contexts; found {len(candidates)}")
                    return candidates[:self.config.few_shot]
        if not candidates:
            raise ValueError(
                f"No eligible few-shot examples for {method}, scale {scale}"
            )
        # Same retained candidate pool and deterministic diverse order for u and d.
        if self.config.selection == "embeddings" and resolve_embeddings:

            def embed(rows: BoardRows) -> Probabilities:
                result = self.api.call(
                    "embeddings",
                    {
                        "model": self.config.embedding_model,
                        "input": [embedding_text(rows)],
                        "encoding_format": "float",
                    },
                )
                vector = np.asarray(result["vectors"][0], dtype=float)
                norm = np.linalg.norm(vector)
                if not np.isfinite(norm) or norm <= 0:
                    raise RequestFailed("Invalid embedding")
                normalized: Probabilities = vector / norm
                return normalized

            query = embed(to_rows(state))
            by_context = {
                example["context"]: example["state"] for example in candidates
            }
            similarity = {
                context: float(embed(rows) @ query)
                for context, rows in by_context.items()
            }
            candidates.sort(key=lambda example: -similarity[example["context"]])
        return self._diverse(candidates, self.config.few_shot)
    def generator_payload(
        self, state: Board, scale: float, resolve_embeddings: bool = True
    ) -> JSONObject:
        """Build a scale-labelled request with a board-size-specific output schema."""
        field = FIELDS[self.config.method]

        def question(rows: BoardRows, value: float) -> JSONObject:
            return {
                "state": rows,
                "goal": task(from_rows(rows)),
                **{field: value},
                "num_subgoals": self.config.num_subgoals,
            }

        def label(example: Example) -> float:
            if self.config.method == "normalized_offset":
                return example["k"] / self.config.offset_denominator
            if field not in example:
                raise KeyError(field)
            value = example.get(field)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Generator example label {field!r} must be numeric")
            return value

        examples = [
            (
                question(example["state"], label(example)),
                {"subgoals": [{"rows": example["subgoal"]}]},
            )
            for example in self.generator_examples(state, scale, resolve_embeddings)
        ]
        h, w = state.shape[:2]
        rows = {
            "type": "array",
            "minItems": h,
            "maxItems": h,
            "items": {"type": "string", "pattern": f"^[#.$*@+\\-]{{{w}}}$"},
        }
        schema = schema_object(
            {
                "subgoals": {
                    "type": "array",
                    "maxItems": self.config.num_subgoals,
                    "items": schema_object({"rows": rows}),
                }
            }
        )
        prompt = (
            self.config.prompts.generator
            + "\n"
            + f"Return each full explicit {h}x{w} board as exactly {h} ASCII rows, "
            + f"each exactly {w} characters, using only # - . * $ @ +. "
            + "Return the complete board in JSON, not a latent vector, coordinate list or description.\n"
            + getattr(self.config.prompts, {"relative_u": "relative", "absolute_d": "absolute"}.get(self.config.method, self.config.method))
        )
        if self.config.method == "normalized_offset":
            prompt += f"\nThe fixed denominator is {self.config.offset_denominator:g} actions."
        prompt += f"\nMaximum controller action cap L={self.config.calibration.action_cap}. Return at most {self.config.num_subgoals} subgoals."
        payload = self._payload(
            "generator",
            prompt,
            examples,
            question(to_rows(state), scale),
            schema=schema,
        )
        if self.config.generator_layout_schema:
            return layout_generator_payload(payload, state, self.config.num_subgoals)
        return payload
    def propose(self, state: Board, scale: float) -> list[Board]:
        """Decode proposals and reject invalid, repeated or unchanged boards."""

        def mock() -> JSONObject:
            candidates = [np.asarray(self.env.next_state(state, a)) for a in range(4)]
            candidates = [s for s in candidates if not np.array_equal(s, state)]
            candidates.sort(key=board_distance)
            rows = [to_rows(candidate) for candidate in candidates[: self.config.num_subgoals]]
            if self.config.generator_layout_schema:
                return {"subgoals": [
                    {f"row_{index:02d}": row for index, row in enumerate(board)} for board in rows
                ]}
            return {"subgoals": [{"rows": board} for board in rows]}

        payload = self.generator_payload(state, scale)
        for attempt in range(self.config.generator_max_attempts):
            result = self._json(payload, mock)
            if (not isinstance(result.get("subgoals"), list)
                    or len(result["subgoals"]) > self.config.num_subgoals):
                raise RequestFailed("Generator returned an invalid subgoals list")
            proposals: list[Board] = []
            seen: set[bytes] = set()
            errors: list[str] = []
            for item in result["subgoals"]:
                try:
                    rows = ([item[f"row_{index:02d}"] for index in range(state.shape[0])]
                            if self.config.generator_layout_schema else item["rows"])
                    proposal = from_rows(rows)
                    validate_subgoal(state, proposal)
                except (TypeError, KeyError, ValueError) as error:
                    self.diagnostics["invalid_proposals"] += 1
                    errors.append(str(error))
                    continue
                key = proposal.argmax(-1).tobytes()
                if np.array_equal(proposal, state):
                    self.diagnostics["self_copies"] += 1
                    errors.append("The proposed board is unchanged")
                elif key in seen:
                    self.diagnostics["duplicate_proposals"] += 1
                    errors.append("The proposed board is duplicated")
                else:
                    seen.add(key)
                    proposals.append(proposal)
            if proposals or attempt + 1 == self.config.generator_max_attempts:
                return proposals
            reason = "; ".join(errors) if errors else "Empty subgoal list"
            scale_instruction = (f"Stay within k={int(scale)} actions." if self.config.method == "fixed"
                                 else "Respect the requested scale.")
            feedback = (
                f"The previous proposal was rejected: {reason}. Return a different complete board "
                "that preserves walls, goals, the number of boxes, and exactly one player. "
                f"{scale_instruction} Do not copy the input."
            )
            payload["input"].extend((text_message("assistant", json.dumps(result)), text_message("user", feedback)))
        return []
