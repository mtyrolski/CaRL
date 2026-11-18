"""Record and visualize the actual explicit-state search, without extra inference."""
from dataclasses import dataclass, field
from typing import Final

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from carl.icl.boards import to_rows
from carl.icl.search import QueueEntry, SearchPath, SearchResult, SearchTransition
from carl.icl.types import BoardRows

ACTION_NAMES: Final = ("UP", "DOWN", "LEFT", "RIGHT")
COLORS: Final = {
    "root": "#3565af",
    "new": "#178b83",
    "reopened": "#8a55a2",
    "duplicate": "#a1a7ae",
    "no_op": "#cdd1d5",
    "rejected": "#e59c38",
    "solved": "#279447",
    "interrupted": "#d94f45",
}


@dataclass
class Expansion:
    """A popped queue entry; queue size is measured immediately after the pop."""

    index: int
    node_id: int
    scale: float
    value: float
    attempted_actions: int
    queue_after_pop: int


@dataclass
class TreeNode:
    """One actual attempt endpoint, including candidates not added to the frontier."""

    node_id: int
    parent_id: int | None
    state: BoardRows
    kind: str
    status: str
    scale: float
    actions: tuple[int, ...]
    path_length: int
    depth: int
    proposal: BoardRows | None = None
    reached: bool = False
    solved: bool = False
    expansions: list[int] = field(default_factory=list)


class SearchTreeRecorder:
    """Observer that records attempt ancestry, queue pops and final accounting.

    Nodes represent attempted paths, not unique boards. Several queue entries can
    expand the same path at different scales. Rejected and duplicate endpoints
    remain visible as leaves and never become expansions.
    """

    def __init__(self) -> None:
        self.nodes: list[TreeNode] = []
        self.expansions: list[Expansion] = []
        self.result: SearchResult | None = None
        self.solution_node: int | None = None
        self._path_ids: dict[int, int] = {}
        self._paths: list[SearchPath] = []

    def _remember(self, path: SearchPath, node_id: int) -> None:
        self._paths.append(path)  # Keep identities alive until recording finishes.
        self._path_ids[id(path)] = node_id

    def on_start(self, root: SearchPath) -> None:
        """Record the root path."""
        self.nodes.append(TreeNode(0, None, to_rows(root.state), "root", "root", 0, (), 0, 0))
        self._remember(root, 0)

    def on_expansion(self, entry: QueueEntry, index: int, actions: int, queue_size: int) -> None:
        """Record an actual non-stale queue pop."""
        node_id = self._path_ids[id(entry.path)]
        self.nodes[node_id].expansions.append(index)
        self.expansions.append(Expansion(index, node_id, -entry.negative_scale, -entry.negative_value, actions, queue_size))

    def on_transition(self, transition: SearchTransition) -> None:
        """Record an actual attempt, retaining even failed candidate paths."""
        parent_id = self._path_ids[id(transition.source)]
        node_id = len(self.nodes)
        node = TreeNode(
            node_id,
            parent_id,
            to_rows(transition.destination.state),
            transition.kind,
            transition.status,
            transition.scale,
            transition.actions,
            len(transition.destination.actions),
            self.nodes[parent_id].depth + 1,
            to_rows(transition.proposal) if transition.proposal is not None else None,
            transition.reached,
            transition.solved,
        )
        self.nodes.append(node)
        if transition.status in {"new", "reopened"}:
            self._remember(transition.destination, node_id)
        if transition.solved:
            self.solution_node = node_id

    def on_finish(self, result: SearchResult) -> None:
        """Retain the solver outcome for inspection and consistency checks."""
        self.result = result
        if result.solved and not result.action_path:
            self.solution_node = 0

    def solution_nodes(self) -> set[int]:
        """Return the exact parent chain of the successful attempt, if any."""
        chain: set[int] = set()
        node_id = self.solution_node
        while node_id is not None:
            chain.add(node_id)
            node_id = self.nodes[node_id].parent_id
        return chain

    def rows(self) -> list[dict[str, object]]:
        """Return compact tabular node metadata; boards stay accessible on nodes."""
        solution = self.solution_nodes()
        return [
            {
                "node": node.node_id,
                "parent": node.parent_id,
                "depth": node.depth,
                "kind": node.kind,
                "status": node.status,
                "scale": node.scale,
                "actions": " ".join(ACTION_NAMES[a] for a in node.actions),
                "path_length": node.path_length,
                "expansions": len(node.expansions),
                "reached": node.reached,
                "solution_path": node.node_id in solution,
            }
            for node in self.nodes
        ]

    def summary(self) -> dict[str, object]:
        """Separate attempt nodes, unique boards, queue expansions and actions."""
        if self.result is None:
            raise ValueError("Search has not finished")
        return {
            "solved": self.result.solved,
            "finish_reason": self.result.finish_reason,
            "attempt_nodes_including_root": len(self.nodes),
            "unique_endpoint_boards": len({tuple(node.state) for node in self.nodes}),
            "expanded_path_nodes": sum(bool(node.expansions) for node in self.nodes),
            "queue_expansions": self.result.expansions,
            "attempted_simulator_actions": self.result.attempted_actions,
            "subgoal_attempts": sum(node.kind == "subgoal" for node in self.nodes),
            "primitive_attempts": sum(node.kind == "primitive" for node in self.nodes),
            "solution_length": len(self.result.action_path) if self.result.solved else None,
            "max_attempt_depth": max(node.depth for node in self.nodes),
            "accepted_subgoals": self.result.accepted,
            "rejected_subgoals": self.result.rejected,
            "reopened_states": self.result.reopened,
            "primitive_successors": self.result.primitive_successors,
            "seconds": self.result.seconds,
            "error": self.result.error,
        }


def tree_figure(tree: SearchTreeRecorder) -> go.Figure:
    """Draw the observed attempt tree, highlighting the exact solution ancestry."""
    children: dict[int, list[int]] = {node.node_id: [] for node in tree.nodes}
    for node in tree.nodes[1:]:
        assert node.parent_id is not None
        children[node.parent_id].append(node.node_id)
    positions: dict[int, float] = {}
    next_leaf = 0

    pending = [(0, False)]
    while pending:
        node_id, visited = pending.pop()
        if not children[node_id]:
            positions[node_id] = float(next_leaf)
            next_leaf += 1
        elif visited:
            values = [positions[child] for child in children[node_id]]
            positions[node_id] = (values[0] + values[-1]) / 2
        else:
            pending.append((node_id, True))
            pending.extend((child, False) for child in reversed(children[node_id]))

    solution = tree.solution_nodes()
    figure = go.Figure()
    for on_solution, color, width in ((False, "#ccd3db", 1), (True, "#279447", 3)):
        xs: list[float | None] = []
        ys: list[float | None] = []
        for node in tree.nodes[1:]:
            parent = node.parent_id
            assert parent is not None
            if (node.node_id in solution and parent in solution) != on_solution:
                continue
            xs.extend((positions[parent], positions[node.node_id], None))
            ys.extend((-tree.nodes[parent].depth, -node.depth, None))
        figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": color, "width": width}, hoverinfo="skip", showlegend=False))
    for status, color in COLORS.items():
        selected = [node for node in tree.nodes if node.status == status]
        if not selected:
            continue
        texts = [
            f"Node {node.node_id} · parent {node.parent_id}<br>{node.kind}: {node.status}"
            f"<br>scale: {node.scale:g} · path actions: {node.path_length}"
            f"<br>segment: {' '.join(ACTION_NAMES[a] for a in node.actions) or '(none)'}"
            f"<br>queue expansions: {len(node.expansions)} · exact target: {node.reached}"
            for node in selected
        ]
        figure.add_trace(
            go.Scatter(
                x=[positions[n.node_id] for n in selected],
                y=[-n.depth for n in selected],
                mode="markers+text",
                text=[str(n.node_id) for n in selected],
                textposition="top center",
                customdata=texts,
                name=status,
                marker={"size": 12, "color": color, "line": {"width": 1, "color": "#263e50"}},
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
    figure.update_layout(
        template="plotly_white",
        title="Observed search attempt tree",
        height=max(420, min(1000, 100 + 70 * (1 + max(n.depth for n in tree.nodes)))),
        xaxis={"visible": False},
        yaxis={"visible": False},
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "b": 20},
    )
    return figure


def expansion_figure(tree: SearchTreeRecorder) -> go.Figure:
    """Show actual action consumption and queue entries after each expansion pop."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    indices = [e.index for e in tree.expansions]
    figure.add_trace(go.Scatter(x=indices, y=[e.attempted_actions for e in tree.expansions], mode="lines+markers", name="Actions before expansion"), secondary_y=False)
    figure.add_trace(go.Scatter(x=indices, y=[e.queue_after_pop for e in tree.expansions], mode="lines+markers", name="Queue entries after pop"), secondary_y=True)
    figure.update_layout(template="plotly_white", title="Search progress at queue pops", xaxis_title="Expansion index", legend={"orientation": "h"})
    figure.update_yaxes(title_text="Simulator actions used", secondary_y=False)
    figure.update_yaxes(title_text="Queue entries (may include stale entries)", secondary_y=True)
    return figure
