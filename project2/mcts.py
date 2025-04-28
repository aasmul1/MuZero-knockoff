import logging
import math
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
                 simulations: int = 5,
                 steps: int = 5, discount: float = 0.997):
        self.c_1 = c_1
        self.c_2 = c_2
        self.simulations = simulations
        self.steps = steps
        self.discount = discount
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
        
        if root_node not in self.search_tree:
            self._expand_node(root_node)

        self.root_nodes.append(root_node)
        self._run_simulations(root_node)
        return self._select_action(root_node, available_actions)

    def _run_simulations(self, root_node: Node):
        num_expanded_nodes = 0

        for simulation in range(self.simulations):
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
                            new_state, reward = self.model.transition(current_node.hidden_state, action_tensor)
                            if isinstance(reward, torch.Tensor):
                                reward = reward.item()
                        except Exception as e:
                            reward = 0
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
                    break

                new_node = self.state_transition_table[(current_node, action)]
                
                if new_node.available_actions is None:
                    self._expand_node(new_node)
                
                trajectory.append((current_node, action))
                current_node = new_node

            trajectory.append((current_node, None))

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


    def _backup(self, trajectory: List[Tuple[Node, Action | None]], leaf_node_value: float) -> None:
        edges = len(trajectory) - 1
        rewards: List[float] = [self.reward_table[edge] for edge in trajectory[0:-1]]

        for k in range(edges):
            cum_reward_towards_leaf_node = sum(
                [self.discount ** tau * rewards[k] for tau in range(0, edges)])
            cumulative_reward = cum_reward_towards_leaf_node + leaf_node_value * self.discount ** (
                    edges - k)

            old_q_value = self.mean_value_table[trajectory[k]]
            old_visit_count = self.visit_count_table[trajectory[k]]

            new_q_value = (old_visit_count * old_q_value + cumulative_reward) / (old_visit_count + 1)
            new_visit_count = old_visit_count + 1

            self.mean_value_table[trajectory[k]] = new_q_value
            self.visit_count_table[trajectory[k]] = new_visit_count


    def _select_action(self, node: Node, available_actions: List[Action]) -> Action:
        total_visit_count = sum(map(lambda a: self.visit_count_table[node, a], available_actions))

        for action in available_actions:
            self.ucb_scores[(node, action)] = self._calculate_ucb_score(node, action,
                                                                        self.visit_count_table[node, action],
                                                                        total_visit_count)

        return max(available_actions,
                   key=lambda a: self.ucb_scores[(node, a)])

    def _calculate_ucb_score(self, node: Node, action: Action, visit_count: int, total_visit_count: int) -> float:
        if total_visit_count == 0:
            return float('inf')
            
        if visit_count == 0:
            return float('inf')
            
        exploitation = self.mean_value_table[(node, action)]
        exploration = self.policy_table[(node, action)] * math.sqrt(
            total_visit_count) / (1 + visit_count) * (
                self.c_1 + math.log((total_visit_count + self.c_2 + 1) / self.c_2))
                
        return exploitation + exploration

    def _expand_node(self, node: Node) -> float:
        if isinstance(node.hidden_state, torch.Tensor) and node.hidden_state.dtype != torch.float32:
            node.hidden_state = node.hidden_state.to(dtype=torch.float32)
            
        policy, value = self.model.predict(node.hidden_state)

        if isinstance(policy, torch.Tensor):
            policy_cpu = policy.detach().cpu()
            if policy_cpu.dtype != torch.float32:
                policy_cpu = policy_cpu.to(dtype=torch.float32)
            policy_dict = {Action(a): p.item() for a, p in zip(range(config.ACTION_SPACE), policy_cpu.flatten())}
            value = value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
        else:
            policy_dict = {Action(a): p for a, p in policy.items()} if isinstance(policy, dict) else policy

        self.search_tree.add(node)
        
        action_keys = list(policy_dict.keys())
        if action_keys and not isinstance(action_keys[0], Action):
            action_keys = [Action(a) for a in action_keys]
        
        node.set_available_actions(action_keys)

        for a in action_keys:
            self.visit_count_table[(node, a)] = 0
            self.mean_value_table[(node, a)] = 0
            self.policy_table[(node, a)] = policy_dict[a]

        return value

    def create_node(self, state) -> Node:
        return next((node for node in self.search_tree if np.array_equal(node.hidden_state, state)), Node(state))
