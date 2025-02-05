import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import offsetbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from carl.environment.sokoban.env import SokobanEnv
from carl.solver.nodes import SearchTreeNode


def layout_tree_horizontal(root: SearchTreeNode,
                           level_height=250,
                           level_width=200) -> dict[SearchTreeNode, tuple[int, int]]:
    def bfs_iterator(root_node: SearchTreeNode):
        queue = [(root_node, 0)]
        while queue:
            node, depth = queue.pop(0)
            yield node, depth
            queue.extend([(child, depth + 1) for child in node.children])

    xs_node = dict(bfs_iterator(root))
    depth2nodes = {depth: [] for depth in xs_node.values()}
    for node, depth in xs_node.items():
        depth2nodes[depth].append(node)

    ys_node = {}
    for node, depth in xs_node.items():
        ys_node[node] = len(depth2nodes[depth]) - 1 - depth2nodes[depth].index(node)

    coordinates = {node: (xs_node[node], ys_node[node]) for node in xs_node}

    pos = {}
    for node, (x, y) in coordinates.items():
        pos[node] = (x * level_width, y * level_height)

    # Set node.y to be a center of its children. Iterate from the right depth to left.
    for depth in sorted(depth2nodes.keys(), reverse=True):
        for node in depth2nodes[depth]:
            if node.children:
                pos[node] = (pos[node][0], np.mean([pos[child][1] for child in node.children]))

    return pos


def create_graph_from_root(node: SearchTreeNode,
                           env: SokobanEnv,
                           level_height: int = 250,
                           level_width: int = 200) -> nx.DiGraph:
    graph = nx.DiGraph()

    pos: dict[SearchTreeNode, tuple[int, int]] = {}

    def add_node_to_graph(node: SearchTreeNode):
        img = env.state_to_repr(node.state, title=f'V(state) = {node.value:.2f}')
        graph.add_node(node, img=img)
        for child in node.children:
            add_node_to_graph(child)
            graph.add_edge(node, child)

    # Create Graph Without Edges
    add_node_to_graph(node)

    # Calculate Nodes Positions layer by layer
    pos = layout_tree_horizontal(node, level_height=level_height, level_width=level_width)

    # Add Positions to Graph
    nx.set_node_attributes(graph, pos, 'pos')

    return graph


def draw_graph(graph: nx.DiGraph, file_path: str = 'graph.png', figsize=(20, 45), node_size=50):
    """Draw the graph with images as node labels and save it to a file."""
    pos = nx.get_node_attributes(graph, 'pos')

    _, ax = plt.subplots(figsize=figsize)    # Create a figure and axes
    nx.draw(graph, pos, ax=ax, with_labels=False, node_size=node_size, arrows=True)

    for node in graph.nodes:
        figure = graph.nodes[node]['img']

        # Convert figure to numpy array
        canvas = FigureCanvas(figure)
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())

        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, zoom=0.5), pos[node], frameon=False)
        ax.add_artist(imagebox)
    plt.savefig(file_path)
    plt.close()
