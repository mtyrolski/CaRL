import pytest

from carl.planners.base import Experience
from carl.planners.base import SearchInfo
from carl.planners.base import Solution
from carl.utils.generator_targets import SubgoalTargetMode
from carl.utils.generator_targets import generate_subgoal_generator_targets
from carl.utils.generator_targets import iter_state_pairs_k_offsets
from tests.utils import assert_equal
from tests.utils import build_dummy_search_tree


def _make_experience(solving_node, subgoal_distance_path=None):
    if subgoal_distance_path is None:
        subgoal_distance_path = []
    search_info = SearchInfo(finished_reason="test", solving_node=solving_node)
    solution = Solution(
        solved=True,
        subgoal_path=[],
        action_path=[],
        subgoal_distance_path=subgoal_distance_path,
    )
    return Experience(solution=solution, search_info=search_info)


def test_generate_subgoal_targets_sliding_window_all_pairs():
    env, solving_node = build_dummy_search_tree(6)
    exp = _make_experience(solving_node)

    targets = generate_subgoal_generator_targets(
        [exp],
        env,
        SubgoalTargetMode.SLIDING_WINDOW,
    )

    path = [0, 1, 2, 3, 4, 5, 6]
    expected = []
    for i in range(len(path) - 1):
        for j in range(i + 1, len(path)):
            expected.append((path[i], path[j]))

    assert_equal(targets, expected)


def test_generate_subgoal_targets_k_offsets():
    env, solving_node = build_dummy_search_tree(6)
    solving_node.next_expand_with_k_generator = 3
    parent = solving_node.parent_node
    assert parent is not None
    parent.next_expand_with_k_generator = 1
    grandparent = parent.parent_node
    assert grandparent is not None
    grandparent.next_expand_with_k_generator = 2
    exp = _make_experience(solving_node)

    targets = generate_subgoal_generator_targets(
        [exp],
        env,
        SubgoalTargetMode.K_OFFSETS,
        k=3,
        offsets=[-1, 1, -2, 2],
    )

    assert_equal(targets, [(2, 6), (4, 6), (1, 6), (5, 6)])


def test_generate_subgoal_targets_k_offsets_fallback_k_list():
    env, solving_node = build_dummy_search_tree(6)
    exp = _make_experience(solving_node, subgoal_distance_path=[2, 1, 3])

    targets = generate_subgoal_generator_targets(
        [exp],
        env,
        SubgoalTargetMode.K_OFFSETS,
        k=3,
        offsets=[-1, 1],
    )

    assert_equal(targets, [(2, 6), (4, 6)])


def test_generate_subgoal_targets_k_offsets_requires_k():
    env, solving_node = build_dummy_search_tree(6)
    exp = _make_experience(solving_node)

    with pytest.raises(ValueError):
        generate_subgoal_generator_targets(
            [exp],
            env,
            SubgoalTargetMode.K_OFFSETS,
        )


def test_iter_state_pairs_k_offsets_radius_one_handcrafted():
    state_path = [0, 1, 2, 3, 4, 5, 6]
    # (position_on_path, k_used) pairs from a hand-crafted hierarchical trajectory
    subgoal_positions = [(2, 3), (3, 3), (6, 3)]

    pairs = list(iter_state_pairs_k_offsets(
        state_path=state_path,
        subgoal_positions=subgoal_positions,
        k=3,
        offsets=[-1, 1],
    ))

    assert_equal(pairs, [(0, 2), (1, 3), (2, 6), (4, 6)])


def test_generate_subgoal_targets_k_offsets_radius_one_from_experience():
    env, solving_node = build_dummy_search_tree(6)
    solving_node.next_expand_with_k_generator = 3
    parent = solving_node.parent_node
    assert parent is not None
    parent.next_expand_with_k_generator = 3
    grandparent = parent.parent_node
    assert grandparent is not None
    grandparent.next_expand_with_k_generator = 3
    exp = _make_experience(solving_node)

    targets = generate_subgoal_generator_targets(
        [exp],
        env,
        SubgoalTargetMode.K_OFFSETS,
        k=3,
        offsets=[-1, 1],
    )

    assert_equal(targets, [(0, 2), (1, 3), (2, 6), (4, 6)])
