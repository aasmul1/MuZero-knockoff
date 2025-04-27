import logging
from typing import Dict, Tuple, List, Set

import numpy as np
import torch

from project2.neural_net import MuZeroNetwork
from project2 import config
from project2.Action import Action
from project2.config import logging_config
from project2.models.super_model import Model
from project2.node import Node
if logging_config["plot_backups"]:
    from project2.utils.visualize_search_tree import visualize_search_tree_with_trajectory


class MCTS:
    def __init__(self, model: Model, c_1: float = 1.25, c_2: float = 19.652,
                 simulations: int = 300,
                 steps: int = 5, discount: float = 0.997):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

        # Used to control the influence of policy relative to the value as nodes are visited more often
        self.c_1 = c_1
        self.c_2 = c_2

        self.simulations = simulations  # Number of simulations to run per search
        self.steps = steps  # The maximum number of hypothetical steps per simulation
        self.discount = discount
        self.model = model

        self.search_tree: Set[Node] = set()
        self.reward_table: Dict[Tuple[Node, Action], float] = dict()  # R(s^(l−1), a^l) = r^l
        self.state_transition_table: Dict[Tuple[Node, Action], Node] = dict()  # S(s^(l−1), a^l) = s^l
        self.visit_count_table: Dict[Tuple[Node, Action], int] = dict()  # N(s^l, a)
        self.mean_value_table: Dict[Tuple[Node, Action], float] = dict()  # Q(s^l, a)
        self.policy_table: Dict[Tuple[Node, Action], float] = dict()  # P(s^l, a) = p^l
        # The policy table stores floats in range [0,1]. All actions of a state sum up to 1, making it a probability distribution over all available actions.

        # For visualizing search tree:
        self.root_nodes: List[Node] = list()  # Store all root nodes use, i.e. the observed states of the environment
        self.ucb_scores: Dict[Tuple[Node, Action], float] = dict()  # Store all calculated UCB scores

    def run_simulations_and_select_action(self, root_node: Node, available_actions: List[int | Action]) -> Action:
        """Run MCTS simulations and select the best action to take from the root node."""
        # Convert int actions to Action objects if needed
        if available_actions and isinstance(available_actions[0], int):
            available_actions = [Action(a) for a in available_actions]
        
        if root_node not in self.search_tree:
            self._expand_node(root_node)
            self.logger.debug(f"Root node expanded")

        self.root_nodes.append(root_node)

        self._run_simulations(root_node)

        return self._select_action(root_node, available_actions)

    def _run_simulations(self, root_node: Node):
        self.logger.info(f"Number of simulations to run is {self.simulations}")
        num_expanded_nodes = 0

        for simulation in range(self.simulations):

            current_node = root_node
            trajectory: List[Tuple[Node, Action | None]] = []
            value = None

            for step in range(self.steps):
                action = self._select_action(current_node, current_node.get_available_actions())

                # Leaf node encountered, expand it
                if (current_node, action) not in self.state_transition_table:
                    # End of simulation:
                    # Compute reward and new state with dynamics function
                    # Store reward and new state in tables
                    # Compute policy and value function for new state with prediction function
                    # Add new node (with value function as attribute?) corresponding to new state to the search tree
                    # Each edge leading out from the newly expanded node is initialized (with the policy)

                    if isinstance(self.model, MuZeroNetwork):
                        # Convert action to tensor if needed
                        if not isinstance(action, torch.Tensor):
                            # Create a long tensor with action number
                            action_tensor = torch.tensor([action.number], dtype=torch.long)
                            # Move to the same device as the hidden state if it's a tensor
                            if isinstance(current_node.hidden_state, torch.Tensor):
                                action_tensor = action_tensor.to(current_node.hidden_state.device)
                        else:
                            action_tensor = action
                        
                        try:
                            new_state, reward = self.model.transition(current_node.hidden_state, action_tensor)
                            
                            # Ensure reward is a scalar
                            if isinstance(reward, torch.Tensor):
                                reward = reward.item()
                        except Exception as e:
                            self.logger.error(f"Error in transition: {e}")
                            # Fallback method if transition fails
                            reward = 0
                            # Create a random state as fallback
                            if isinstance(current_node.hidden_state, torch.Tensor):
                                device = current_node.hidden_state.device
                                shape = current_node.hidden_state.shape
                                new_state = torch.zeros(shape, device=device)
                            else:
                                new_state = np.zeros_like(current_node.hidden_state)
                    else:
                        new_state, reward = self.model.transition(current_node.hidden_state, action)
                    
                    new_node = Node(new_state)
                    self.reward_table[(current_node, action)] = reward
                    self.state_transition_table[(current_node, action)] = new_node
                    value = self._expand_node(new_node)
                    trajectory.append((current_node, action))
                    num_expanded_nodes += 1
                    self.logger.debug(f"Expanded node! Reward {reward}, value {value}")
                    break  # Only one expansion per simulation

                new_node = self.state_transition_table[(current_node, action)]
                trajectory.append((current_node, action))
                current_node = new_node

            self.logger.debug(f"Completed steps of simulation {simulation}. {step + 1} steps taken.")

            trajectory.append((current_node, None))  # Add last node visited/expanded to end of trajectory

            if len(trajectory) <= 1 or not value:
                self.logger.debug(f"Skipping backup. Only a single node in trajectory, i.e., first node expanded")
                continue  # Only a single node in trajectory, i.e., first node expanded

            if logging_config["plot_backups"]:
                visualize_search_tree_with_trajectory(trajectory=trajectory, mean_value_table=self.mean_value_table,
                                                      reward_table=self.reward_table, policy_table=self.policy_table,
                                                      state_transition_table=self.state_transition_table,
                                                      ucb_scores=self.ucb_scores,
                                                      visit_count_table=self.visit_count_table,
                                                      file_name=f"before_backup-{simulation}.png",
                                                      search_tree=self.search_tree)

            self._backup(trajectory, value)  # TODO As of now, only do backup when encountered unexpanded node

            if logging_config["plot_backups"]:
                # Update stored UCB scores.
                temp_tot_visit_counts = dict()
                for n_a_tuple in trajectory[:-1]:  # Do not include leaf node of trajectory
                    if n_a_tuple[0] not in temp_tot_visit_counts:
                        temp_tot_visit_counts[n_a_tuple[0]] = sum(
                            [visit_count for key, visit_count in self.visit_count_table.items() if
                             key[0] == n_a_tuple[0]])

                    self.ucb_scores[n_a_tuple] = self._calculate_ucb_score(node=n_a_tuple[0], action=n_a_tuple[1],
                                                                           visit_count=self.visit_count_table[
                                                                               n_a_tuple],
                                                                           total_visit_count=temp_tot_visit_counts[
                                                                               n_a_tuple[0]])

                visualize_search_tree_with_trajectory(trajectory=trajectory, mean_value_table=self.mean_value_table,
                                                      reward_table=self.reward_table, policy_table=self.policy_table,
                                                      state_transition_table=self.state_transition_table,
                                                      ucb_scores=self.ucb_scores,
                                                      visit_count_table=self.visit_count_table,
                                                      file_name=f"after_backup-{simulation}.png",
                                                      search_tree=self.search_tree)

            self.logger.info(f"Back up performed of simulation {simulation} on trajectory of length {len(trajectory)}")

        self.logger.info(f"Done with simulations. {num_expanded_nodes} nodes expanded.")

    def _backup(self, trajectory: List[Tuple[Node, Action | None]], leaf_node_value: float) -> None:
        edges = len(trajectory) - 1
        rewards: List[float] = [self.reward_table[edge] for edge in trajectory[
                                                                    0:-1]]  # Not including last element of trajectory, which is only the leaf node

        self.logger.debug(f"Performing backup... Rewards {rewards}")

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
            self.logger.debug(
                f"Updated edge. Old Q {round(old_q_value, 2)}, New Q {round(new_q_value, 2)}, Old V {old_visit_count}, New V {new_visit_count}, Leaf node value {leaf_node_value}, Cum Reward towards leaf {cum_reward_towards_leaf_node}, Cum Reward {cumulative_reward}")

    def _select_action(self, node: Node, available_actions: List[Action]) -> Action:
        total_visit_count = sum(map(lambda a: self.visit_count_table[node, a], available_actions))

        for action in available_actions:
            self.ucb_scores[(node, action)] = self._calculate_ucb_score(node, action,
                                                                        self.visit_count_table[node, action],
                                                                        total_visit_count)

        return max(available_actions,
                   key=lambda a: self.ucb_scores[(node, a)])

    def _calculate_ucb_score(self, node: Node, action: Action, visit_count: int, total_visit_count: int) -> float:
        # Same formula as in MuZero
        # return self.mean_value_table[(node, action)] + self.policy_table[(node, action)] * math.sqrt(
        #     total_visit_count) / (1 + visit_count) * (
        #         self.c_1 + math.log((total_visit_count + self.c_2 + 1) / self.c_2))
        # TODO Remove this after finished troubleshooting backprop
        return self.mean_value_table[(node, action)]

    def _expand_node(self, node: Node) -> float:
        # Ensure node's hidden state is float32 if it's a tensor
        if isinstance(node.hidden_state, torch.Tensor) and node.hidden_state.dtype != torch.float32:
            node.hidden_state = node.hidden_state.to(dtype=torch.float32)
            
        policy, value = self.model.predict(node.hidden_state)

        # Convert policy to dictionary format based on what type it is
        if isinstance(policy, torch.Tensor):
            # Convert tensor to CPU before using it
            policy_cpu = policy.detach().cpu()
            # Ensure policy is float32
            if policy_cpu.dtype != torch.float32:
                policy_cpu = policy_cpu.to(dtype=torch.float32)
            policy_dict = {Action(a): p.item() for a, p in zip(range(config.ACTION_SPACE), policy_cpu.flatten())}
            value = value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
        else:
            policy_dict = {Action(a): p for a, p in policy.items()} if isinstance(policy, dict) else policy

        self.search_tree.add(node)
        
        # Convert policy dict keys to Action objects if they aren't already
        action_keys = list(policy_dict.keys())
        if action_keys and not isinstance(action_keys[0], Action):
            action_keys = [Action(a) for a in action_keys]
        
        node.set_available_actions(action_keys)

        self.logger.debug(
            f"Policy: {policy_dict}, Value: {value}. Available actions in expansion: {action_keys}")

        for a in action_keys:
            # Initialize edge
            self.visit_count_table[(node, a)] = 0
            self.mean_value_table[(node, a)] = 0
            self.policy_table[(node, a)] = policy_dict[a]

        return value

    # TODO This has to be worked around later
    def create_node(self, state) -> Node:
        # Return node if in search tree, else create new
        return next((node for node in self.search_tree if np.array_equal(node.hidden_state, state)), Node(state))
