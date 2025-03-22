import copy
import random
from typing import List

import numpy as np

from project2.game import Game


class Node:
    def __init__(self, sim_game: Game, parent: "Node" = None, action: int = None):
        self.value = None  # TODO Correct init?
        self.actions = None  # TODO Correct init?
        self.parent = parent  # Where we came from with the action
        self.sim_game = sim_game  # TODO Not deepcopy?
        self.action = action  # The action that got us to this node
        self.children: List[Node] = []
        self.visits = 0
        self.reward = 0

    def copy(self):
        return copy.deepcopy(self)

    def follow_tree_policy(self):
        if self.visits == 0:  # The node has never been expanded
            return self

        best_child = self.get_best_child()

        while True:
            if best_child.visits == 0 or best_child.sim_game.is_terminal_state() or not best_child.children:
                break
            best_child = best_child.get_best_child()

        return best_child

    def get_best_child(self):
        assert self.children, "No children found"

        # Child scores of all children that has been expanded
        children_scores = [(child, child.get_ucb_score()) for child in self.children if child.visits > 0]

        if not children_scores:
            return random.choice(self.children)  # If none of the children have been visited, i.e. expanded

        max_value = max(children_scores, key=lambda x: x[1])[1]
        best_children = [child for child, score in children_scores if score == max_value]

        # Choose random child with best UCB score, but if none are visited, choose randomly
        # TODO This does not choose a child that has not been visited if some child has been visited, but it should!
        return random.choice(best_children) if best_children else random.choice(self.children)

    def expand(self):
        if self.sim_game.is_terminal_state():
            return self

        # enten metode fra game_state_manager eller catch.py
        self.actions = self.sim_game.get_legal_actions()
        if not self.actions:
            print("No actions found")
            return self

        # TODO Apply or step method?
        children = [Node(self.sim_game.copy().apply(action), parent=self, action=action)
                    for action in self.actions if action is not None]

        random.shuffle(children)
        self.children = children
        return self.get_best_child()

    # Upper Confidence Bound
    def get_ucb_score(self):

        c = 1.41

        if self.visits == 0:
            return float('inf')

        top_node = self
        if top_node.parent:
            top_node = top_node.parent

        return (self.reward / self.visits) + c * np.sqrt(np.log(top_node.visits) / self.visits)

    def backpropagate(self, reward) -> None:
        self.visits += 1
        self.value += (reward - self.value) / self.visits
        if self.parent:
            self.parent.backpropagate(reward)

    def get_distribution(self):
        children_action_indices = []
        for child in self.children:
            children_action_indices.append(child.action)
            # TODO Previously .get_action_index(). Should action be a class? Is action_index unique for the environment, ref. next lines?

        distribution = np.zeros(len(self.sim_game.get_actions()))
        distribution[children_action_indices] = [child.visits for child in self.children]
        distribution /= distribution.sum()  # Normalizing distribution
        return distribution
