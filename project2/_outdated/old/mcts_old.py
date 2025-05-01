import copy
import random
from typing import Dict

from project2.core.game import Game
from project2.old.node_old import NodeOld

MCTS_POLICY_EXPLORE = 100


def mcts(mytree: "MCTSOld"):
    for i in range(MCTS_POLICY_EXPLORE):
        mytree.explore()

    next_tree, next_action = mytree.next()

    next_tree.detatch_parent()  # TODO Not declared anywhere

    return next_tree, next_action


class MCTSOld:
    def __init__(self, sim_game: Game, current_node: NodeOld):
        self.sim_game: Game = sim_game.copy()
        self.done = False  # TODO Correct init?
        self.children: Dict[int, NodeOld] = {}  # key: action, value: child that the action leads to
        # TODO Correct init?

        if current_node:
            self.current_node = current_node
        else:
            NodeOld(self.sim_game)  # TODO Assign to current_node?

    def explore(self):

        current_node = self
        while current_node.children:
            child = current_node.children
            max_u = max(c.get_ucb_score() for c in child.values())
            actions = [a for a, c in child.items() if c.get_ucb_score() == max_u]
            assert len(actions), f"Error zero length: {max_u}"
            action = random.choice(actions)
            current_node = child[action]

        if current_node.visits < 1:
            current_node.reward = current_node.reward + current_node.rollout()
        else:
            current_node.create_child()
            if current_node.children:
                current_node = random.choice(current_node.children)
            current_node.reward = current_node.reward + current_node.rollout()

        current_node.visits += 1

        while current_node.parent:
            child_reward = current_node.reward
            current_node = current_node.parent
            current_node.visits += 1
            current_node.reward += child_reward

    def rollout(self):

        if self.done:
            return 0

        v = 0
        done = False
        new_game = copy.deepcopy(self.sim_game)
        while not done:
            # Get a random action
            action = new_game.action_space.sample()
            observation, reward, done, _ = new_game.step(action)
            v = v + reward
            if done:
                new_game.reset()
                new_game.close()
                break
        return v

    def next(self):
        assert not self.done, "Game has ended"
        assert self.children, "No children found and game hasn\'t ended"

        # TODO Only single child in children?
        child = self.children

        max_visits = max(node.visits for node in child.values())

        max_children = [c for a, c in child.items() if c == max_visits]

        if len(max_children) == 0:
            print("Error zero length ", max_visits)

        max_child = random.choice(max_children)

        return max_child, max_child.action
