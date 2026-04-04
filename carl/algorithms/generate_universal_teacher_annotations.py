from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger

from carl.algorithms.algorithm import Algorithm
from carl.dataloader.universal_generator_types import UniversalTeacherAnnotation
from carl.planners.base import Experience
from carl.solver.nodes import SearchTreeNode
from carl.utils.aliases import State


def _copy_state(state: State) -> State:
    if isinstance(state, np.ndarray):
        return state.copy()
    return state


def _hashable_state(state: State) -> tuple[int, ...] | tuple[str, ...]:
    if isinstance(state, np.ndarray):
        return tuple(map(int, state.flatten()))
    return tuple(state)


class GenerateUniversalTeacherAnnotations(Algorithm):
    """Build proposition-only teacher annotations from saved solve traces.

    Preferred source is proposal event logs stored in `Experience.search_info.proposal_events`.
    If absent (older traces), falls back to accepted subgoals reconstructed from the stored search tree.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        input_glob: str = "solved_problems_*.joblib",
        max_experiences: int | None = None,
        max_annotations: int | None = None,
        include_unsolved: bool = False,
        include_rejected_from_events: bool = True,
        prefer_proposal_events: bool = True,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.input_glob = input_glob
        self.max_experiences = max_experiences
        self.max_annotations = max_annotations
        self.include_unsolved = include_unsolved
        self.include_rejected_from_events = include_rejected_from_events
        self.prefer_proposal_events = prefer_proposal_events

    def _iter_files(self) -> list[Path]:
        if self.input_path.is_dir():
            return sorted(self.input_path.glob(self.input_glob))
        if self.input_path.is_file():
            return [self.input_path]
        return sorted(Path().glob(str(self.input_path)))

    def _flatten_experiences(self, item: object) -> list[Experience]:
        out: list[Experience] = []
        stack: list[object] = [item]
        while stack:
            cur = stack.pop()
            if isinstance(cur, Experience):
                out.append(cur)
            elif isinstance(cur, list):
                stack.extend(cur)
        return out

    def _load_experiences(self) -> list[Experience]:
        experiences: list[Experience] = []
        for file in self._iter_files():
            logger.info(f"Loading experiences from {file}")
            experiences.extend(self._flatten_experiences(joblib.load(file)))
            if self.max_experiences is not None and len(experiences) >= self.max_experiences:
                break
        if self.max_experiences is not None:
            experiences = experiences[:self.max_experiences]
        logger.info(f"Loaded {len(experiences)} experiences")
        return experiences

    def _annotations_from_events(self, exp: Experience) -> list[UniversalTeacherAnnotation]:
        annotations: list[UniversalTeacherAnnotation] = []
        proposal_events = getattr(exp.search_info, "proposal_events", [])
        for event in proposal_events:
            validator_accept = bool(event.get("validator_accept", False))
            validator_reject = bool(event.get("validator_reject", False))
            if not self.include_rejected_from_events and not validator_accept:
                continue
            current_state = event.get("current_state")
            proposition_state = event.get("proposed_state")
            if current_state is None or proposition_state is None:
                continue
            ann = UniversalTeacherAnnotation(
                current_state=_copy_state(current_state),
                proposition_state=_copy_state(proposition_state),
                validator_accept=validator_accept,
                validator_reject=validator_reject,
                reached=bool(event.get("reached", False)),
                source="proposal_event",
                metadata={
                    "generator_mode": getattr(exp.search_info, "generator_mode", None),
                    "proposal_rank": event.get("proposal_rank"),
                    "proposal_confidence": event.get("proposal_confidence"),
                    "duplicate": event.get("duplicate", False),
                    "validator_low_level_nodes_visited": event.get("validator_low_level_nodes_visited"),
                },
            )
            annotations.append(ann)
        return annotations

    def _annotations_from_search_tree(self, exp: Experience) -> list[UniversalTeacherAnnotation]:
        root = getattr(exp.search_info, "search_tree", None)
        if root is None:
            return []

        queue: deque[SearchTreeNode] = deque([root])
        annotations: list[UniversalTeacherAnnotation] = []
        seen_edges: set[tuple[tuple[int, ...] | tuple[str, ...], tuple[int, ...] | tuple[str, ...]]] = set()

        while queue:
            node = queue.popleft()
            for child in node.children:
                queue.append(child)
                # Skip planner scaffolding edges (e.g., root frontier seed nodes).
                if child.low_level_path is None:
                    continue

                parent_key = _hashable_state(node.state)
                child_key = _hashable_state(child.state)
                edge_key = (parent_key, child_key)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                annotations.append(
                    UniversalTeacherAnnotation(
                        current_state=_copy_state(node.state),
                        proposition_state=_copy_state(child.state),
                        validator_accept=True,
                        validator_reject=False,
                        reached=bool(child.is_on_solving_path),
                        source="search_tree_accepted_fallback",
                        metadata={
                            "generator_mode": getattr(exp.search_info, "generator_mode", None),
                            "node_depth": child.metadata.get("depth"),
                        },
                    )
                )
        return annotations

    def _build_annotations(self, experiences: list[Experience]) -> list[UniversalTeacherAnnotation]:
        annotations: list[UniversalTeacherAnnotation] = []
        for exp in experiences:
            if not self.include_unsolved and not exp.solution.solved:
                continue

            current: list[UniversalTeacherAnnotation] = []
            proposal_events = getattr(exp.search_info, "proposal_events", [])
            if self.prefer_proposal_events and proposal_events:
                current = self._annotations_from_events(exp)
            if not current:
                current = self._annotations_from_search_tree(exp)

            annotations.extend(current)
            if self.max_annotations is not None and len(annotations) >= self.max_annotations:
                break

        if self.max_annotations is not None:
            annotations = annotations[:self.max_annotations]
        return annotations

    def run(self) -> None:
        experiences = self._load_experiences()
        annotations = self._build_annotations(experiences)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "meta": {
                "input_path": str(self.input_path),
                "input_glob": self.input_glob,
                "max_experiences": self.max_experiences,
                "max_annotations": self.max_annotations,
                "include_unsolved": self.include_unsolved,
                "include_rejected_from_events": self.include_rejected_from_events,
                "prefer_proposal_events": self.prefer_proposal_events,
                "num_experiences_loaded": len(experiences),
                "num_annotations": len(annotations),
            },
            "annotations": [asdict(ann) for ann in annotations],
        }
        joblib.dump(payload, self.output_path)
        logger.success(f"Wrote {len(annotations)} universal teacher annotations to {self.output_path}")
