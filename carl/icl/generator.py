"""Drop-in ICL proposal component for adaptive and universal CaRL planners."""

import json
from pathlib import Path
from typing import Any, Final

import numpy as np

from carl.icl.api import BudgetExceeded, CachedOpenAI, RequestFailed
from carl.icl.boards import embedding_text, from_rows, to_rows, validate_subgoal
from carl.icl.config import ModelSettings
from carl.inference_components.component import InferenceComponent, RawComponent
from carl.solver.nodes import GeneratedSubgoal, SearchTreeNode

PROMPT: Final[
    str
] = """Propose intermediate Sokoban states following the demonstrations.
Symbols: # wall, - floor, . goal, $ box, * box on goal, @ player, + player on goal.
The task is to place every box on a goal. Preserve all walls, goal squares, the
number of boxes and exactly one player. Boxes can be pushed, never pulled.
Propose distinct, reachable states different from the current state, useful for
solving the puzzle. Return only JSON with a subgoals array of objects with rows.
When k is supplied it is a primitive-action horizon: demonstrations show states
k trajectory actions later, and the controller gets at most k actions to reach
each proposal. It is not a success probability, percentile or embedding scale.
Without k, propose states reachable within the separate controller action cap.
Return up to the requested number of subgoals. An empty array is allowed."""


class OpenAISubgoalGenerator(InferenceComponent):
    def __init__(
        self,
        env: Any,
        model: str,
        demonstrations_path: str | Path,
        api: CachedOpenAI,
        *,
        planner: str = "adaptive",
        generator_k_list: list[int] | None = None,
        num_subgoals: int = 2,
        examples_per_request: int = 4,
        selection: str = "direct",
        embedding_model: str = "text-embedding-3-small",
        max_output_tokens: int = 2048,
        reasoning_effort: str | None = "low",
        controller_cap: int = 8,
        max_expansions: int = 100,
        transport: ModelSettings | None = None,
    ) -> None:
        if planner not in {"adaptive", "universal"} or selection not in {"direct", "embeddings"}:
            raise ValueError("Unknown planner or demonstration selection")
        if (
            min(
                num_subgoals,
                examples_per_request,
                max_output_tokens,
                controller_cap,
                max_expansions,
            )
            < 1
        ):
            raise ValueError("Generator limits must be positive")
        self.env, self.model, self.api = env, model, api
        self.transport = transport
        self.planner, self.selection, self.embedding_model = planner, selection, embedding_model
        self.mode = "bank" if planner == "adaptive" else "propositional_universal"
        self.generator_k_list = list(generator_k_list or [8, 4, 1]) if planner == "adaptive" else []
        if any(
            not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in self.generator_k_list
        ) or len(set(self.generator_k_list)) != len(self.generator_k_list):
            raise ValueError("Horizons must be distinct positive integers")
        self.num_subgoals, self.examples_per_request = num_subgoals, examples_per_request
        self.max_output_tokens, self.reasoning_effort = max_output_tokens, reasoning_effort
        self.controller_cap, self.max_expansions = controller_cap, max_expansions
        self.demonstrations = json.loads(Path(demonstrations_path).read_text(encoding="utf-8"))[
            "demonstrations"
        ]
        if not self.demonstrations:
            raise ValueError("At least one demonstration is required")
        for example in self.demonstrations:
            validate_subgoal(from_rows(example["state"]), from_rows(example["subgoal"]))
            if (
                not isinstance(example.get("k"), int)
                or isinstance(example["k"], bool)
                or example["k"] < 1
            ):
                raise ValueError("Demonstrations need a positive trajectory horizon k")
        for k in self.generator_k_list:
            if not any(example["k"] == k for example in self.demonstrations):
                raise ValueError(f"No demonstrations for horizon k={k}")
        self.metrics = {
            "expansions": 0,
            "decoded": 0,
            "invalid": 0,
            "self_copies": 0,
            "duplicates": 0,
            "malformed": 0,
        }

    def construct_network(self) -> None:
        """API client is lazy; no model download or key needed to prepare prompts."""

    def get_network(self) -> RawComponent:
        return {}

    def _examples(
        self, rows: list[str], k: int | None, resolve_embeddings: bool
    ) -> list[dict[str, Any]]:
        candidates = [example for example in self.demonstrations if k is None or example["k"] == k]
        if self.selection == "embeddings" and resolve_embeddings:
            # Cache each board separately so demonstrations are embedded only once.
            texts = [embedding_text(rows)] + [
                embedding_text(example["state"]) for example in candidates
            ]
            vectors = []
            for text in texts:
                result = self.api.call(
                    "embeddings",
                    {"model": self.embedding_model, "input": [text], "encoding_format": "float"},
                )
                vector = np.asarray(result["vectors"][0], dtype=float)
                norm = np.linalg.norm(vector)
                if not np.isfinite(vector).all() or norm == 0:
                    raise RequestFailed("Invalid embedding vector")
                vectors.append(vector / norm)
            similarities = [float(vector @ vectors[0]) for vector in vectors[1:]]
            order = sorted(range(len(candidates)), key=lambda i: (-similarities[i], i))
            candidates = [candidates[i] for i in order]
        return candidates[: self.examples_per_request]

    def build_payload(
        self, node: SearchTreeNode, resolve_embeddings: bool = True
    ) -> dict[str, Any]:
        rows = to_rows(node.state)
        k = node.next_expand_with_k_generator
        if self.planner == "adaptive" and k not in self.generator_k_list:
            raise ValueError("Adaptive planner must supply a configured horizon k")
        if self.planner == "universal" and k is not None:
            raise ValueError("Universal planner must not supply k")
        height, width = len(rows), len(rows[0])
        schema = {
            "type": "object",
            "properties": {
                "subgoals": {
                    "type": "array",
                    "maxItems": self.num_subgoals,
                    "items": {
                        "type": "object",
                        "properties": {
                            "rows": {
                                "type": "array",
                                "minItems": height,
                                "maxItems": height,
                                "items": {"type": "string", "pattern": f"^[#.$*@+\\-]{{{width}}}$"},
                            }
                        },
                        "required": ["rows"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["subgoals"],
            "additionalProperties": False,
        }

        def query(board: list[str]) -> str:
            request: dict[str, Any] = {
                "state": board,
                "num_subgoals": self.num_subgoals,
                "controller_cap": k or self.controller_cap,
            }
            if k is not None:
                request["k"] = k
            return json.dumps(request)

        output_format = self.transport.response_format if self.transport else "json_schema"
        prompt = PROMPT
        if output_format != "json_schema":
            prompt += "\nReturn only JSON matching this schema: " + json.dumps(schema)
        messages: list[dict[str, Any]] = [{"role": "developer", "content": prompt}]
        for example in self._examples(rows, k, resolve_embeddings):
            messages.extend(
                [
                    {"role": "user", "content": query(example["state"])},
                    {
                        "role": "assistant",
                        "content": json.dumps({"subgoals": [{"rows": example["subgoal"]}]}),
                    },
                ]
            )
        # A developer-message boundary after all demonstrations preserves the
        # reusable prefix when the dynamic board at the end changes (GPT-6+).
        use_explicit_cache = self.model.startswith("gpt-6") and (
            self.transport is None or self.transport.provider == "openai"
        )
        if use_explicit_cache:
            messages.append(
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Apply the demonstrations to the next board.",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                }
            )
        messages.append({"role": "user", "content": query(rows)})
        text_format: dict[str, Any] = {"type": output_format}
        if output_format == "json_schema":
            text_format.update(name="sokoban_subgoals", strict=True, schema=schema)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "store": False,
            "max_output_tokens": self.max_output_tokens,
            "text": {"format": text_format},
        }
        if self.transport is not None:
            payload["metadata"] = {"component": "generator"}
            if self.transport.temperature is not None:
                payload["temperature"] = self.transport.temperature
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if use_explicit_cache:
            payload["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
        return payload

    def _mock(self, state: np.ndarray | str) -> dict[str, Any]:
        # Legal one-step successors, explicitly not an LLM or a k-step generator.
        if not isinstance(state, np.ndarray):
            raise ValueError("Only Sokoban array states are supported")
        successors, seen = [], {tuple(state.argmax(-1).flat)}
        for action in (3, 2, 1, 0):
            after = self.env.next_state(state, action)
            key = tuple(after.argmax(-1).flat)
            if key not in seen:
                seen.add(key)
                successors.append({"rows": to_rows(after)})
        return {
            "output_text": json.dumps({"subgoals": successors[: self.num_subgoals]}),
            "usage": {},
            "model": "mock_legal_successors",
        }

    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        if self.metrics["expansions"] >= self.max_expansions:
            raise BudgetExceeded("Run high-level expansion limit reached")
        self.metrics["expansions"] += 1
        payload = self.build_payload(node)
        result = self.api.call("responses", payload, mock=lambda: self._mock(node.state))
        try:
            decoded = json.loads(result["output_text"])
            if (
                not isinstance(decoded, dict)
                or set(decoded) != {"subgoals"}
                or not isinstance(decoded["subgoals"], list)
            ):
                raise ValueError("Malformed subgoal response")
            if len(decoded["subgoals"]) > self.num_subgoals:
                raise ValueError("Too many subgoals")
        except (ValueError, TypeError):
            self.metrics["malformed"] += 1
            return []
        proposals, seen = [], set()
        for rank, proposal in enumerate(decoded["subgoals"]):
            self.metrics["decoded"] += 1
            try:
                if not isinstance(proposal, dict) or set(proposal) != {"rows"}:
                    raise ValueError("Malformed proposal")
                subgoal = from_rows(proposal["rows"])
                validate_subgoal(node.state, subgoal)
            except (ValueError, TypeError):
                self.metrics["invalid"] += 1
                continue
            if np.array_equal(subgoal, node.state):
                self.metrics["self_copies"] += 1
                continue
            key = tuple(subgoal.argmax(-1).flat)
            if key in seen:
                self.metrics["duplicates"] += 1
                continue
            seen.add(key)
            proposals.append(
                GeneratedSubgoal(subgoal, {"proposal_rank": rank, "generator_mode": self.mode})
            )
        return proposals
