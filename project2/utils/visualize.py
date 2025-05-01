import logging
import os
from typing import Set, Dict, Tuple, List

import networkx as nx
import pygraphviz as pgv
from PIL import Image

from project2.Action import Action
from project2.node import Node


def visualize_search_tree(search_tree: Set[Node],
                          root_nodes: List[Node],
                          reward_table: Dict[Tuple[Node, Action], float],
                          state_transition_table: Dict[Tuple[Node, Action], Node],
                          visit_count_table: Dict[Tuple[Node, Action], int],
                          mean_value_table: Dict[Tuple[Node, Action], float],
                          policy_table: Dict[Tuple[Node, Action], float],
                          ucb_scores: Dict[Tuple[Node, Action], float],
                          file_name: str = "plot.png"):
    logger = logging.getLogger(__name__)
    logger.info("Started creating plot of graph...")

    g = pgv.AGraph(directed=True)
    g.add_nodes_from(search_tree)

    edge_weights = list()

    for node in search_tree:
        node_action_tuples = [(key[0], key[1]) for key in state_transition_table.keys() if key[0] == node]

        # node_action_weights = {
        #     n_a_tuple: f"R {reward_table[n_a_tuple]}, V {visit_count_table[n_a_tuple]}, Q {round(mean_value_table[n_a_tuple], 2)}, P {round(policy_table[n_a_tuple], 2)}"
        #     for n_a_tuple in node_action_tuples}

        node_action_weights = {
            n_a_tuple: f"{round(ucb_scores[n_a_tuple], 2)}"
            for n_a_tuple in node_action_tuples}

        edge_weights.extend([(key[0], value, node_action_weights[(key[0], key[1])]) for key, value in
                             state_transition_table.items() if (key[0], key[1]) in node_action_tuples])

        edges = [(key[0], value) for key, value in state_transition_table.items() if
                 (key[0], key[1]) in node_action_tuples]

        g.add_edges_from(edges)

    for edge_weight in edge_weights:
        g.get_edge(edge_weight[0], edge_weight[1]).attr['label'] = edge_weight[2]

    _mark_longest_path_nodes(g, root_nodes)

    # g.graph_attr['nodesep'] = '100'  # Increase space between nodes
    # g.graph_attr['ranksep'] = '5'  # Increase space between ranks
    g.node_attr['fontsize'] = '5'  # Set font size for better readability
    g.edge_attr['fontsize'] = '7'  # Adjust edge label font size
    g.graph_attr['splines'] = 'true'  # Use smooth edge curves
    # g.graph_attr['width'] = '8'  # Width of the graph (in inches)
    # g.graph_attr['height'] = '6'  # Height of the graph (in inches)
    g.graph_attr['overlap'] = 'false'  # Ensure no node overlap

    g.layout(prog="twopi")

    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    g.draw(f"plots/{file_name}")

    logger.info(f"Finished creating plot of graph! Saved at plots/{file_name}")


def _mark_longest_path_nodes(g, root_nodes):
    nx_g = nx.nx_agraph.from_agraph(g)

    for root_node in root_nodes:
        if root_node == root_nodes[0]:
            g.get_node(root_node).attr['color'] = 'blue'
        elif root_node == root_nodes[-1]:
            g.get_node(root_node).attr['color'] = 'red'
        else:
            g.get_node(root_node).attr['color'] = 'green'

        path_lengths = nx.single_source_shortest_path_length(nx_g, g.get_node(root_node))
        first_longest_path_node = max(path_lengths, key=lambda n: path_lengths[n])
        longest_path_length = path_lengths[first_longest_path_node]
        longest_path_nodes = [n for n in path_lengths if path_lengths[n] == longest_path_length]

        # for node in longest_path_nodes:
        #     g.get_node(node).attr['color'] = 'yellow'


def visualize_search_tree_with_trajectory(trajectory: List[Tuple[Node, Action | None]],
                                          search_tree: Set[Node],
                                          reward_table: Dict[Tuple[Node, Action], float],
                                          state_transition_table: Dict[Tuple[Node, Action], Node],
                                          visit_count_table: Dict[Tuple[Node, Action], int],
                                          mean_value_table: Dict[Tuple[Node, Action], float],
                                          policy_table: Dict[Tuple[Node, Action], float],
                                          ucb_scores: Dict[Tuple[Node, Action], float],
                                          file_name: str):
    # trajectory_search_tree = {node_action_tuple[0] for node_action_tuple in trajectory}
    trajectory_nodes = [node_action_tuple[0] for node_action_tuple in trajectory]
    visualize_search_tree(search_tree=search_tree, ucb_scores=ucb_scores,
                          visit_count_table=visit_count_table, policy_table=policy_table, reward_table=reward_table,
                          state_transition_table=state_transition_table, mean_value_table=mean_value_table,
                          root_nodes=trajectory_nodes, file_name=file_name)


def visualize_played_game(states: List, plot_file_name="played_game.png", gif_file_name="played_game.png"):
    logger = logging.getLogger(__name__)
    logger.info(f"Generating visualization of played game...")

    # Make plot
    g = pgv.AGraph(directed=True)
    g.add_nodes_from(states)

    edges = list()
    for i in range(len(states) - 1):
        edges.append((states[i], states[i + 1]))

    g.add_edges_from(edges)

    g.node_attr['fontsize'] = '5'  # Set font size for better readability
    g.edge_attr['fontsize'] = '7'  # Adjust edge label font size
    g.graph_attr['splines'] = 'true'  # Use smooth edge curves
    g.graph_attr['overlap'] = 'false'  # Ensure no node overlap

    g.layout(prog="twopi")
    os.makedirs(f"plots", exist_ok=True)
    g.draw(f"plots/{plot_file_name}")

    logger.info(f"Finished creating graph of played game! plots/{plot_file_name}")

    # Make gif
    image_paths = list()
    for i, state in enumerate(states):
        g = pgv.AGraph(directed=True)
        g.add_node(state)
        g.layout(prog="twopi")
        g.draw(f"plots/frame_{i}.png")
        image_paths.append(f"plots/frame_{i}.png")

    images = [Image.open(file_path) for file_path in image_paths]
    images[0].save(
        f"plots/{gif_file_name}",
        save_all=True,
        append_images=images[1:],
        duration=0.0001,  # TODO This does not work
        loop=0
    )

    logger.info(f"Finished creating gif of played game! plots/{gif_file_name}")
