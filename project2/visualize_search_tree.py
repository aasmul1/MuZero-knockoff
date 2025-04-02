from typing import Set, Dict, Tuple

import networkx as nx
import pygraphviz as pgv

from project2.Action import Action
from project2.node import Node


def visualize_search_tree(search_tree: Set[Node],
                          root_nodes: Set[Node],
                          reward_table: Dict[Tuple[Node, Action], float],
                          state_transition_table: Dict[Tuple[Node, Action], Node],
                          visit_count_table: Dict[Tuple[Node, Action], int],
                          mean_value_table: Dict[Tuple[Node, Action], float],
                          policy_table: Dict[Tuple[Node, Action], float],
                          ucb_scores: Dict[Tuple[Node, Action], float]):
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
    g.draw("plot.png")


def _mark_longest_path_nodes(g, root_nodes):
    nx_g = nx.nx_agraph.from_agraph(g)

    for root_node in root_nodes:
        g.get_node(root_node).attr['color'] = 'blue'

        path_lengths = nx.single_source_shortest_path_length(nx_g, g.get_node(root_node))
        first_longest_path_node = max(path_lengths, key=lambda n: path_lengths[n])
        longest_path_length = path_lengths[first_longest_path_node]
        longest_path_nodes = [n for n in path_lengths if path_lengths[n] == longest_path_length]

        for node in longest_path_nodes:
            g.get_node(node).attr['color'] = 'green'
