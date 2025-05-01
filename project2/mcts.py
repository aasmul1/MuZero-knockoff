import math
import time
from typing import Dict, Tuple, List, Set

import numpy as np
import torch

from project2.Action import Action
from project2.config import logging_config
from project2.models.super_model import Model
from project2.neural_net import MuZeroNetwork
from project2.node import Node

if logging_config["plot_backups"]:
    from project2.utils.visualize import visualize_search_tree_with_trajectory


class MCTS:
    def __init__(self, model: Model, config):
        self.config = config
        self.c_1 = self.config.MCTS_C1
        self.c_2 = self.config.MCTS_C2
        self.simulations = self.config.SIMULATIONS
        self.steps = self.config.STEPS
        self.discount = self.config.MCTS_DISCOUNT
        self.model = model
        self.search_tree: Set[Node] = set()
        self.reward_table: Dict[Tuple[Node, Action], float] = dict()
        self.state_transition_table: Dict[Tuple[Node, Action], Node] = dict()
        self.visit_count_table: Dict[Tuple[Node, Action], int] = dict()
        self.mean_value_table: Dict[Tuple[Node, Action], float] = dict()
        self.policy_table: Dict[Tuple[Node, Action], float] = dict()
        self.root_nodes: List[Node] = list()
        self.ucb_scores: Dict[Tuple[Node, Action], float] = dict()

    def run_simulations_and_select_action(self, root_node: Node, available_actions: List[int | Action]) -> Action:
        if available_actions and isinstance(available_actions[0], int):
            available_actions = [Action(a) for a in available_actions]

        # Ensure the root node's hidden state is float32
        if isinstance(root_node.hidden_state, torch.Tensor) and root_node.hidden_state.dtype != torch.float32:
            root_node.hidden_state = root_node.hidden_state.to(dtype=torch.float32)

        if root_node not in self.search_tree:
            self._expand_node(root_node)

        self.root_nodes.append(root_node)
        self._run_simulations(root_node)
        return self._select_action(root_node, available_actions)

    def _run_simulations(self, root_node: Node):
        num_expanded_nodes = 0
        start_time = time.time()
        max_simulation_time = 120

        for simulation in range(self.simulations):
            if time.time() - start_time > max_simulation_time:
                self.logger.warning(f"MCTS simulations timed out after {simulation} iterations")
                break

            current_node = root_node
            trajectory: List[Tuple[Node, Action | None]] = []
            value = None

            for step in range(self.steps):
                if current_node.available_actions is None:
                    self._expand_node(current_node)

                action = self._select_action(current_node, current_node.get_available_actions())

                if (current_node, action) not in self.state_transition_table:
                    if isinstance(self.model, MuZeroNetwork):
                        if not isinstance(action, torch.Tensor):
                            action_tensor = torch.tensor([action.number], dtype=torch.long)
                            if isinstance(current_node.hidden_state, torch.Tensor):
                                action_tensor = action_tensor.to(current_node.hidden_state.device)
                        else:
                            action_tensor = action

                        try:
                            # Ensure hidden_state is float32
                            if isinstance(current_node.hidden_state,
                                          torch.Tensor) and current_node.hidden_state.dtype != torch.float32:
                                current_node.hidden_state = current_node.hidden_state.to(dtype=torch.float32)

                            new_state, reward = self.model.transition(current_node.hidden_state, action_tensor)

                            # Convert reward to float32 if it's a tensor
                            if isinstance(reward, torch.Tensor):
                                if reward.dtype != torch.float32:
                                    reward = reward.to(dtype=torch.float32)
                                reward = float(reward.item())
                        except Exception as e:
                            reward = 0.0  # Use 0.0 instead of 0
                            if isinstance(current_node.hidden_state, torch.Tensor):
                                device = current_node.hidden_state.device
                                shape = current_node.hidden_state.shape
                                new_state = torch.zeros(shape, device=device,
                                                        dtype=torch.float32)  # Explicitly set dtype
                            else:
                                new_state = np.zeros_like(current_node.hidden_state,
                                                          dtype=np.float32)  # Explicitly set dtype
                    else:
                        new_state, reward = self.model.transition(current_node.hidden_state, action)
                        # Convert reward to float if it's numpy float
                        if isinstance(reward, (np.floating, np.float64, np.float32)):
                            reward = float(reward)

                    new_node = Node(new_state)
                    self.reward_table[(current_node, action)] = float(reward)  # Ensure Python float
                    self.state_transition_table[(current_node, action)] = new_node
                    value = self._expand_node(new_node)
                    trajectory.append((current_node, action))
                    num_expanded_nodes += 1
                    break

                new_node = self.state_transition_table[(current_node, action)]

                if new_node.available_actions is None:
                    self._expand_node(new_node)

                trajectory.append((current_node, action))
                current_node = new_node

            trajectory.append((current_node, None))

            # If value is still None after traversal, we need to compute it for the final node
            if value is None and current_node.hidden_state is not None:
                # Try to get the value from the last node
                try:
                    if current_node.available_actions is None:
                        value = self._expand_node(current_node)
                    else:
                        # Predict value from the current node's hidden state
                        _, predicted_value = self.model.predict(current_node.hidden_state)
                        if isinstance(predicted_value, torch.Tensor):
                            if predicted_value.dtype != torch.float32:
                                predicted_value = predicted_value.to(dtype=torch.float32)
                            value = float(predicted_value.detach().cpu().item())
                        else:
                            value = float(predicted_value) if predicted_value is not None else 0.0
                except Exception:
                    # If prediction fails, use default value
                    value = 0.0

            # Ensure value is never None before calling _backup
            if value is None:
                value = 0.0

            if logging_config["plot_backups"]:
                visualize_search_tree_with_trajectory(trajectory=trajectory, mean_value_table=self.mean_value_table,
                                                      reward_table=self.reward_table, policy_table=self.policy_table,
                                                      state_transition_table=self.state_transition_table,
                                                      ucb_scores=self.ucb_scores,
                                                      visit_count_table=self.visit_count_table,
                                                      file_name=f"before_backup-{simulation}.png",
                                                      search_tree=self.search_tree)

            self._backup(trajectory, value)

            if logging_config["plot_backups"]:
                temp_tot_visit_counts = dict()
                for n_a_tuple in trajectory[:-1]:
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

    def _backup(self, trajectory: List[Tuple[Node, Action | None]], leaf_node_value: float | None) -> None:
        edges = len(trajectory) - 1
        rewards: List[float] = [self.reward_table[edge] for edge in trajectory[0:-1]]

        # Handle None value
        if leaf_node_value is None:
            leaf_node_value = 0.0  # Default to 0.0 if we got a None value

        # Convert leaf_node_value to float32
        if isinstance(leaf_node_value, torch.Tensor):
            if leaf_node_value.dtype == torch.float64:
                leaf_node_value = leaf_node_value.to(torch.float32)
            leaf_node_value = leaf_node_value.item()
        leaf_node_value = float(leaf_node_value)  # Ensure Python float, not numpy.float64

        for k in range(edges):
            cum_reward_towards_leaf_node = float(sum(
                [float(self.discount ** tau) * float(rewards[k]) for tau in range(0, edges)]))

            # Avoid using float64 in calculations
            discount_power = float(self.discount ** (edges - k))
            cumulative_reward = cum_reward_towards_leaf_node + leaf_node_value * discount_power

            old_q_value = self.mean_value_table[trajectory[k]]
            old_visit_count = self.visit_count_table[trajectory[k]]

            new_q_value = float((old_visit_count * old_q_value + cumulative_reward) / (old_visit_count + 1))
            new_visit_count = old_visit_count + 1

            self.mean_value_table[trajectory[k]] = new_q_value
            self.visit_count_table[trajectory[k]] = new_visit_count

    def _select_action(self, node: Node, available_actions: List[Action]) -> Action:
        # Ensure we have at least one available action
        if not available_actions:
            # If no actions available, return a default action
            return Action(0)  # Create a default action

        # Initialize missing entries in visit_count_table
        for action in available_actions:
            if (node, action) not in self.visit_count_table:
                self.visit_count_table[(node, action)] = 0
                self.mean_value_table[(node, action)] = 0.0
                self.policy_table[(node, action)] = 1.0 / len(available_actions)

        total_visit_count = sum(map(lambda a: self.visit_count_table.get((node, a), 0), available_actions))

        for action in available_actions:
            self.ucb_scores[(node, action)] = self._calculate_ucb_score(node, action,
                                                                        self.visit_count_table.get((node, action), 0),
                                                                        total_visit_count)

        return max(available_actions,
                   key=lambda a: self.ucb_scores.get((node, a), float('-inf')))

    def _calculate_ucb_score(self, node: Node, action: Action, visit_count: int, total_visit_count: int) -> float:
        if total_visit_count == 0:
            return float('inf')

        if visit_count == 0:
            return float('inf')

        # Ensure we have entries in our tables
        if (node, action) not in self.mean_value_table:
            self.mean_value_table[(node, action)] = 0.0

        if (node, action) not in self.policy_table:
            # Default to uniform policy if missing
            self.policy_table[(node, action)] = 1.0 / len(
                node.get_available_actions()) if node.available_actions else 0.5

        exploitation = float(self.mean_value_table[(node, action)])

        # Calculate exploration term with explicit float conversions
        policy_value = float(self.policy_table[(node, action)])
        total_sqrt = float(math.sqrt(total_visit_count))
        c1 = float(self.c_1)
        c2 = float(self.c_2)

        exploration = policy_value * total_sqrt / (1.0 + float(visit_count)) * (
                c1 + float(math.log((float(total_visit_count) + c2 + 1.0) / c2)))

        return float(exploitation + exploration)

    def _expand_node(self, node: Node) -> float:
        if isinstance(node.hidden_state, torch.Tensor) and node.hidden_state.dtype != torch.float32:
            node.hidden_state = node.hidden_state.to(dtype=torch.float32)

        try:
            policy, value = self.model.predict(node.hidden_state)
        except Exception as e:
            # If prediction fails, use default values
            policy = {Action(a): 1.0 / self.config.ACTION_SPACE for a in range(self.config.ACTION_SPACE)}
            value = 0.0

        if isinstance(policy, torch.Tensor):
            policy_cpu = policy.detach().cpu()
            if policy_cpu.dtype != torch.float32:
                policy_cpu = policy_cpu.to(dtype=torch.float32)
            policy_dict = {Action(a): float(p.item()) for a, p in
                           zip(range(self.config.ACTION_SPACE), policy_cpu.flatten())}

            # Handle the value tensor properly
            if isinstance(value, torch.Tensor):
                if value.dtype != torch.float32:
                    value = value.to(dtype=torch.float32)
                value = float(value.detach().cpu().item())
        else:
            policy_dict = {Action(a): float(p) for a, p in policy.items()} if isinstance(policy, dict) else policy
            if isinstance(value, (np.floating, np.float64, np.float32)):
                value = float(value)  # Convert numpy float to Python float
            elif value is None:
                value = 0.0  # Default value if None

        self.search_tree.add(node)

        # Ensure policy_dict is valid
        if not policy_dict or not isinstance(policy_dict, dict):
            policy_dict = {Action(a): 1.0 / self.config.ACTION_SPACE for a in range(self.config.ACTION_SPACE)}

        action_keys = list(policy_dict.keys())
        if action_keys and not isinstance(action_keys[0], Action):
            action_keys = [Action(a) for a in action_keys]

        node.set_available_actions(action_keys)

        for a in action_keys:
            self.visit_count_table[(node, a)] = 0
            self.mean_value_table[(node, a)] = 0.0  # Explicit float
            self.policy_table[(node, a)] = float(policy_dict[a])  # Ensure Python float

        # Ensure we always return a valid float
        if value is None:
            value = 0.0

        return float(value)

    def create_node(self, state) -> Node:
        return next((node for node in self.search_tree if np.array_equal(node.hidden_state, state)), Node(state))
