from __future__ import annotations

from pathlib import Path

import joblib

from carl.algorithms.generate_universal_teacher_annotations import GenerateUniversalTeacherAnnotations
from carl.planners.base import Experience
from carl.planners.base import SearchInfo
from carl.planners.base import Solution
from carl.solver.nodes import SearchTreeNode


def test_generate_universal_teacher_annotations_from_search_tree_fallback(tmp_path: Path):
    root = SearchTreeNode("s0", 0.0, None, None, metadata={"depth": -1})
    scaffold = SearchTreeNode("s0", 0.0, None, root, next_expand_with_k_generator=8, metadata={"depth": 0})
    accepted = SearchTreeNode("s1", 0.1, [1], scaffold, next_expand_with_k_generator=8, metadata={"depth": 1})
    accepted.is_on_solving_path = True
    exp = Experience(
        solution=Solution(solved=True, subgoal_path=["s1"], action_path=[1], subgoal_distance_path=[8]),
        search_info=SearchInfo(finished_reason="solved", search_tree=root, solving_node=accepted),
    )

    in_file = tmp_path / "solved.joblib"
    out_file = tmp_path / "teacher.joblib"
    joblib.dump([exp], in_file)

    alg = GenerateUniversalTeacherAnnotations(
        input_path=str(in_file),
        output_path=str(out_file),
        prefer_proposal_events=True,
    )
    alg.run()

    payload = joblib.load(out_file)
    assert payload["meta"]["num_annotations"] >= 1
    ann = payload["annotations"][0]
    assert ann["current_state"] == "s0"
    assert ann["proposition_state"] == "s1"
    assert ann["validator_accept"] is True
