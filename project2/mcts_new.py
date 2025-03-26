import math
from typing import Dict, Tuple, List, Set

from project2.Action import Action
from project2.models.super_model import Model
from project2.node_new import NodeNew


class MCTSNew:
    def __init__(self, model: Model, available_actions: List[Action], c_1: float = 1.25, c_2: float = 19.652,
                 simulations: int = 800,
                 steps: int = 5, discount: float = 0.997):
        # Used to control the influence of policy relative to the value as nodes are visited more often
        self.c_1 = c_1
        self.c_2 = c_2

        self.simulations = simulations  # Number of simulations to run per search
        self.steps = steps  # The maximum number of hypothetical steps per simulation
        self.discount = discount
        self.model = model
        self.available_actions = available_actions  # TODO Do we know this?

        self.search_tree: Set[NodeNew] = set()
        self.reward_table: Dict[Tuple[NodeNew, Action], float] = dict()  # R(s^(l−1), a^l) = r^l
        self.state_transition_table: Dict[Tuple[NodeNew, Action], NodeNew] = dict()  # S(s^(l−1), a^l) = s^l
        self.visit_count_table: Dict[Tuple[NodeNew, Action], int] = dict()  # N(s^l, a)
        self.mean_value_table: Dict[Tuple[NodeNew, Action], float] = dict()  # Q(s^l, a)
        self.policy_table: Dict[Tuple[NodeNew, Action], float] = dict()  # P(s^l, a) = p^l
        # The policy table stores floats in range [0,1]. All actions of a state sum up to 1, making it a probability distribution over all available actions.

    def run_simulations_and_select_action(self, root_node: NodeNew) -> Action:
        if root_node not in self.search_tree:
            self._expand_node(root_node, self.available_actions)

        self._run_simulations(root_node)

        return self._select_action(root_node, self.available_actions)

    def _run_simulations(self, root_node: NodeNew):

        for simulation in range(self.simulations):

            current_node = root_node
            trajectory: List[Tuple[NodeNew, Action | None]] = []
            value = None

            for step in range(self.steps):
                action = self._select_action(current_node, self.available_actions)

                # Leaf node encountered, expand it
                if (current_node, action) not in self.state_transition_table:
                    # End of simulation:
                    # Compute reward and new state with dynamics function
                    # Store reward and new state in tables
                    # Compute policy and value function for new state with prediction function
                    # Add new node (with value function as attribute?) corresponding to new state to the search tree
                    # Each edge leading out from the newly expanded node is initialized (with the policy)

                    reward, new_state = self.model.transition(current_node, action)
                    new_node = NodeNew(new_state)
                    self.reward_table[(current_node, action)] = reward
                    self.state_transition_table[(current_node, action)] = new_node
                    value = self._expand_node(new_node, self.available_actions)
                    trajectory.append((current_node, action))
                    break  # Only one expansion per simulation

                new_node = self.state_transition_table[(current_node, action)]
                trajectory.append((current_node, action))
                current_node = new_node

            trajectory.append((current_node, None))  # Add last node visited/expanded to end of trajectory

            if len(trajectory) <= 1 or not value:
                return  # Only a single node in trajectory, i.e., first node expanded

            self._backup(trajectory, value)  # TODO As of now, only do backup when encountered unexpanded node

    def _backup(self, trajectory: List[Tuple[NodeNew, Action | None]], leaf_node_value: float):
        edges = len(trajectory) - 1
        rewards: List[float] = [self.reward_table[edge] for edge in trajectory[
                                                                    0:-1]]  # Not including last element of trajectory, which is only the leaf node

        # Generate cumulative discounted rewards for updating edge values. k = 0 is the first node in the trajectory
        # The cum rewards are bootstrapped from the value of the last node in trajectory (which comes from the prediction function)
        for k in range(edges):
            cum_reward_towards_leaf_node = sum(
                [self.discount ** tau * rewards[k] for tau in range(0, edges)])  # Summation in the G^k formula
            cumulative_reward = cum_reward_towards_leaf_node + leaf_node_value * self.discount ** (
                    edges - k)  # G^k formula

            old_q_value = self.mean_value_table[trajectory[k]]
            old_visit_count = self.visit_count_table[trajectory[k]]

            new_q_value = (old_visit_count * old_q_value + cumulative_reward) / (old_visit_count + 1)
            new_visit_count = old_visit_count + 1

            self.mean_value_table[trajectory[k]] = new_q_value
            self.visit_count_table[trajectory[k]] = new_visit_count

    def _select_action(self, node: NodeNew, available_actions: List):
        total_visit_count = sum(map(lambda a: a.visit_count, available_actions))

        return max(available_actions,
                   key=lambda a: self._calculate_ucb_score(node, a, a.visit_count, total_visit_count))

    def _calculate_ucb_score(self, node: NodeNew, action: Action, visit_count: int, total_visit_count: int):
        # Same formula as in MuZero
        return self.mean_value_table[(node, action)] + self.policy_table[(node, action)] * math.sqrt(
            total_visit_count) / (1 + visit_count) * (
                self.c_1 + math.log((total_visit_count + self.c_2 + 1) / self.c_2))

    def _expand_node(self, node: NodeNew, available_actions) -> float:
        policy, value = self.model.predict(node.hidden_state)  # TODO Assuming policy returned is a dict
        self.search_tree.add(node)

        for a in available_actions:
            # Initialize edge
            self.visit_count_table[(node, a)] = 0
            self.mean_value_table[(node, a)] = 0
            self.policy_table[(node, a)] = policy[a]

        return value
