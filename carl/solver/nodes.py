import hashlib
import queue
from collections import namedtuple

import numpy as np
from loguru import logger

from carl.environment.sokoban.env import printable_sokoban_state


class SearchTreeNode:
    """
    A high-level node in the search tree.

    Stores information about the environment state and search metadata,
    such as its value_function, depth in the tree, list of child nodes, parent node
    and the low-level path that leads to it from its parent.
    """
    def __init__(
        self,
        state,
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

        self.children = []
        self.is_on_solving_path = False

        # None only if root or before adding to the planner
        self.next_expand_with_k_generator = next_expand_with_k_generator

        self.metadata = metadata if metadata is not None else {}

        if parent_node is not None and 'depth' in parent_node.metadata:
            self.metadata['depth'] = parent_node.metadata['depth'] + 1

        if self.parent_node is not None:
            self.parent_node.children.append(self)


def copy_solving_node(solving_node: SearchTreeNode):
    reversed_node_path = []
    current_node = solving_node    # last node of the search (last state)

    while current_node is not None:
        # Create a raw nodes
        reversed_node_path.append(
            SearchTreeNode(
                state=current_node.state.copy(),
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
    new_experiences = []

    for experience in experiences:
        solution: dict[str, bool | list[np.ndarray]] = {
            'solved': True,
            'subgoal_path': [],
            'action_path': [],
            'subgoal_distance_path': [],
        }

        search_info = {
            'solved': True,
            'solving_node': copy_solving_node(experience[1]['solving_node']),
        }

        new_experiences.append((solution, search_info))
    return new_experiences


# Result of a validation.
ValidationResult = namedtuple(
    'ValidationResult',
    ['is_valid', 'is_solved', 'path', 'low_level_nodes_visited', 'achieved_state'],
)

# Result of a generation.
GeneratedSubgoal = namedtuple('GeneratedSubgoal', ['state', 'generation_metadata'])
GeneratedAction = namedtuple('GeneratedAction', ['action', 'generation_metadata'])


def hash_numpy_array(array):
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


def hash_numpy_arrays(arrays: list):
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
        self.counter = 0
        self.queue = queue.PriorityQueue()

    def put(self, data, key):
        self.queue.put((key, self.counter, data))
        self.counter += 1

    def get(self):
        if self.queue.empty():
            return None

        return self.queue.get()[-1]

    def empty(self):
        return self.queue.empty()


def get_solving_path_data(solving_node, include_state_path=True, env=None):
    subgoal_path = []
    action_path = []
    values = []
    state_path = None

    current_node = solving_node

    while current_node.parent_node is not None:
        subgoal_path.append(current_node.state)
        values.append(current_node.value)
        current_node.is_on_solving_path = True

        if current_node.low_level_path is not None:
            action_path.append(current_node.low_level_path)

        current_node = current_node.parent_node

    # subgoal_path.append(current_node.state)
    current_node.is_on_solving_path = True

    subgoal_path.reverse()
    action_path.reverse()
    action_path = flatten(action_path)

    if include_state_path:
        env.restore_full_state_from_np_array_version(subgoal_path[0])
        state_path = [env.get_state()]

        for action in action_path:
            state, _, _, _ = env.step(action)
            state_path.append(state)

    return subgoal_path, action_path, values, state_path


def flatten(l):
    return [item for sublist in l for item in sublist]


def print_search_tree(root_node, max_depth=4):
    cnt = 0

    root_node.metadata['print_id'] = 0
    nodes_to_print = [root_node]

    for i in range(max_depth):
        print(f'--- LEVEL {i} ---\n')

        next_layer = []

        for node in nodes_to_print:
            print(f'Id: {node.metadata["print_id"]}')
            print(printable_sokoban_state(node.state))
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
