import hashlib
import queue
from collections import namedtuple

import numpy as np
from loguru import logger
from typing import Protocol

from carl.utils.aliases import State
from carl.utils.loggers import log_error_and_raise

PriorityKey = float | tuple[int, float]


class SearchTreeNode:
    """
    A high-level node in the search tree.

    Stores information about the environment state and search metadata,
    such as its value_function, depth in the tree, list of child nodes, parent node
    and the low-level path that leads to it from its parent.
    """
    def __init__(
        self,
        state: State,
        value,
        low_level_path,
        parent_node,
        next_expand_with_k_generator: int | None = None,
        metadata=None,
    ):
        self.state = state
        self.value = value
        self.low_level_path = low_level_path
        self.parent_node = parent_node

        self.children: list["SearchTreeNode"] = []
        self.is_on_solving_path = False

        # None only if root or before adding to the planner
        self.next_expand_with_k_generator = next_expand_with_k_generator

        self.metadata = metadata if metadata is not None else {}

        if parent_node is not None and 'depth' in parent_node.metadata:
            self.metadata['depth'] = parent_node.metadata['depth'] + 1

        if self.parent_node is not None:
            self.parent_node.children.append(self)

    def __repr__(self):
        state_shape = self.state.shape if isinstance(self.state, np.ndarray) else f'({len(self.state)},)'
        return f'SearchTreeNode(state.shape={state_shape}, value={round(self.value, 2)}, next_expand_k={self.next_expand_with_k_generator})'

def copy_solving_node(solving_node: SearchTreeNode):
    reversed_node_path = []
    current_node = solving_node    # last node of the search (last state)

    while current_node is not None:
        # Create a raw nodes
        reversed_node_path.append(
            SearchTreeNode(
                state=current_node.state.copy() if isinstance(current_node.state, np.ndarray) else current_node.state,
                value=current_node.value,
                low_level_path=[x for x in current_node.low_level_path]
                if current_node.low_level_path is not None else None,
                parent_node=None,
                next_expand_with_k_generator=current_node.next_expand_with_k_generator,
            ))

        current_node = current_node.parent_node

    node_path = reversed_node_path[::-1]

    # link parents
    for i in range(1, len(node_path)):
        node_path[i].parent_node = node_path[i - 1]

    return node_path[-1]    # return the last node of the search (last state)


def prune_experiences(experiences):
    from carl.planners.base import Experience, SearchInfo, Solution
    from carl.planners.base import Experience, SearchInfo, Solution
    new_experiences = []

    for experience in experiences:
        solution = Solution(
            solved=True,
            subgoal_path=[],
            action_path=[],
            subgoal_distance_path=[],
        )
        solving_node = experience.search_info.solving_node
        search_info = SearchInfo(
            finished_reason=experience.search_info.finished_reason,
            solving_node=copy_solving_node(solving_node) if solving_node is not None else None,
        )

        new_experiences.append(Experience(solution=solution, search_info=search_info))
    return new_experiences


# Result of a validation.
ValidationResult = namedtuple(
    'ValidationResult',
    ['is_valid', 'is_solved', 'path', 'low_level_nodes_visited', 'achieved_state'],
)

# Result of a generation.
GeneratedSubgoal = namedtuple('GeneratedSubgoal', ['state', 'generation_metadata'])
GeneratedAction = namedtuple('GeneratedAction', ['action', 'generation_metadata'])


def hash_numpy_array(array: np.ndarray) -> str:
    """
    Hashes a NumPy array using SHA-256 and returns the hexadecimal hash value.

    Parameters:
    - array: NumPy array to be hashed.

    Returns:
    - Hexadecimal string representing the hash of the array.
    Priority queue that uses a counter to ensure unique keys, sorts the elements as lowest-first.

    """
    # Convert the array to bytes
    array_bytes = array.tobytes()

    # Create a sha256 hash object
    hash_obj = hashlib.sha256()

    # Update the hash object with the array bytes
    hash_obj.update(array_bytes)

    # Return the hexadecimal digest of the hash object
    return hash_obj.hexdigest()


def hash_numpy_arrays(arrays: list[np.ndarray]) -> str:
    """
    Hashes a list of NumPy arrays using SHA-256 and returns the hexadecimal hash value.

    Parameters:
    - arrays: List of NumPy arrays to be hashed.

    Returns:
    - Hexadecimal string representing the hash of the arrays.
    """
    # Create a sha256 hash object
    hash_obj = hashlib.sha256()

    # Iterate over the arrays
    for array in arrays:
        # Convert the array to bytes
        array_bytes = array.tobytes()

        # Update the hash object with the array bytes
        hash_obj.update(array_bytes)

    # Return the hexadecimal digest of the hash object
    return hash_obj.hexdigest()


class SafePriorityQueue:
    """Priority queue that uses a counter to ensure unique keys, sorts the elements as lowest-first."""
    def __init__(self):
        super().__init__()
        self.counter: int = 0
        self.queue: queue.PriorityQueue[tuple[PriorityKey, int, SearchTreeNode]] = queue.PriorityQueue()

    def put(self, data: SearchTreeNode, key: PriorityKey) -> None:
        self.queue.put((key, self.counter, data))
        self.counter += 1

    def get(self) -> SearchTreeNode | None:
        if self.queue.empty():
            return None

        return self.queue.get()[-1]

    def empty(self) -> bool:
        return self.queue.empty()
    
    def __len__(self) -> int:
        return self.queue.qsize()
    
    def size(self) -> int:
        """Returns the size of the queue."""
        return len(self)

def get_solving_path_data(
    solving_node: SearchTreeNode,
    include_state_path: bool = True,
    env: "EnvWithRestore | None" = None,
) -> tuple[list[State], list[int], list[float], list[int], list[State] | None]:
    """
    Extracts the solving path data from the solving node.
    Returns the subgoal path, action path, values, k_used, and state path if include_state_path is True.
    If env is None and include_state_path is True, raises an error.
    """
    subgoal_path: list[State] = []
    action_path: list[int] = []
    values: list[float] = []
    k_used: list[int] = []
    if env is None and include_state_path:
        log_error_and_raise(
            'Environment is required to include state path in the solving path data.'
        )
    state_path = None

    current_node = solving_node

    while current_node.parent_node is not None:
        subgoal_path.append(current_node.state)
        values.append(current_node.value)
        if current_node.next_expand_with_k_generator is not None:
            k_used.append(current_node.next_expand_with_k_generator)
        current_node.is_on_solving_path = True

        if current_node.low_level_path is not None:
            action_path.append(current_node.low_level_path)

        current_node = current_node.parent_node

    current_node.is_on_solving_path = True

    subgoal_path.reverse()
    values.reverse()
    k_used.reverse()
    action_path.reverse()
    action_path = flatten(action_path)

    if include_state_path:
        assert env is not None
        env.restore_full_state_from_np_array_version(current_node.state)
        state_path = [env.get_state()]

        for action in action_path:
            state, _, _, _ = env.step(action)
            state_path.append(state)

    return subgoal_path, action_path, values, k_used, state_path


class EnvWithRestore(Protocol):
    def restore_full_state_from_np_array_version(self, state: State) -> None:
        ...

    def get_state(self) -> State:
        ...

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        ...


def flatten(lists):
    return [item for sublist in lists for item in sublist]


def print_search_tree(root_node, max_depth=4):
    cnt = 0

    root_node.metadata['print_id'] = 0
    nodes_to_print = [root_node]

    for i in range(max_depth):
        print(f'--- LEVEL {i} ---\n')

        next_layer = []

        for node in nodes_to_print:
            print(f'Id: {node.metadata["print_id"]}')
            print(f'value: {node.value}, path: {node.low_level_path}, metadata: {node.metadata}')

            if node.parent_node is not None:
                print(f'parent_id: {node.parent_node.metadata["print_id"]}')

            children = []

            for child in node.children:
                cnt += 1
                child.metadata['print_id'] = cnt
                children.append(child.metadata['print_id'])
                next_layer.append(child)

            print(f'children_id: {children}\n')

        nodes_to_print = next_layer


def hashable_state(state):
    if isinstance(state, np.ndarray):
        return tuple(state.flatten())
    else:
        return state


def prune_search_tree_from_solving_node(solving_node: SearchTreeNode):
    if len(solving_node.children) > 0:
        logger.error('Solving node should not have children')
        raise ValueError('Solving node should not have children')

    current_node = solving_node
    while current_node.parent_node is not None:
        current_node.parent_node.children = [current_node]
        current_node = current_node.parent_node


def dfs(tree_node: SearchTreeNode):
    for child_node in tree_node.children:
        dfs(child_node)
    print(tree_node.state)


def get_root_from_solving_node(solving_node: SearchTreeNode):
    current_node = solving_node
    while current_node.parent_node is not None:
        current_node = current_node.parent_node
    return current_node


def high_level_tree_size(tree_node: SearchTreeNode):
    size = 1
    for child_node in tree_node.children:
        size += high_level_tree_size(child_node)
    return size


def low_level_tree_size(tree_node: SearchTreeNode):
    # Includes also intermediate low level states
    size = 1
    for child_node in tree_node.children:
        size += low_level_tree_size(child_node)

    if tree_node.low_level_path is not None:
        size += len(tree_node.low_level_path)

    return size


def search_tree_stats(tree_node: SearchTreeNode):
    return f"""
    High level tree size: {high_level_tree_size(tree_node)}
    Low level tree size: {low_level_tree_size(tree_node)}
    """


def get_hash_of_solved_board(solution_and_search_info: tuple[dict, dict]) -> str:
    solution, _ = solution_and_search_info
    if not solution['solved']:
        raise ValueError('Board is not solved')

    subgoal_path = solution['subgoal_path']

    return hash_numpy_arrays(subgoal_path)


def get_unique_count(solution_and_search_infos: list[tuple[dict, dict]]) -> int:
    hashes = list(map(get_hash_of_solved_board, solution_and_search_infos))
    return len(set(hashes))
