# train_muzero.py

from project2 import config
from project2.mcts import MCTS
from project2.neural_net_manager import NeuralNetManager
from project2.replay_buffer import ReplayBuffer
from project2.buffer_game import Game as BufferGame
from project2.games.catch import CatchGame
from project2.node import Node
from project2.Action import Action  

import torch
import matplotlib.pyplot as plt


def main():
    # Initialize managers
    network_manager = NeuralNetManager(config)
    replay_buffer = ReplayBuffer(config)

    num_iterations = 50
    episodes_per_iteration = 5
    batches_per_iteration = 10

    all_losses = []


    for iteration in range(num_iterations):
        print(f"=== Iteration {iteration} ===")

        # --- Self-Play ---
        for episode in range(episodes_per_iteration):
            game = CatchGame()
            mcts = MCTS(network_manager.model)

            buffer_game = BufferGame()

            observation = game.state_to_observation()
            observation_tensor = torch.tensor(observation).unsqueeze(0)
            network_output = network_manager.initial_inference(observation_tensor)
            root_node = Node(network_output.hidden_state)

            while not game.game_over:
                legal_actions = [Action(a) for a in game.get_legal_actions()]   # <-- wrap them
                action = mcts.run_simulations_and_select_action(root_node, legal_actions)

                buffer_game.observations.append(observation_tensor.clone())
                buffer_game.actions.append(action)   # no need to wrap again

                next_state, reward, done = game.step(action.number)   # <-- use action.number 

                buffer_game.rewards.append(reward)

                # Update observation and root for next move
                observation = game.state_to_observation()
                observation_tensor = torch.tensor(observation).unsqueeze(0)
                network_output = network_manager.initial_inference(observation_tensor)
                root_node = Node(network_output.hidden_state)
                root_node.available_actions = legal_actions


            replay_buffer.save_game(buffer_game)

        # --- Training ---
        for batch in range(batches_per_iteration):
            batch_data = replay_buffer.sample_batch()

            # First: filter where actions are non-empty
            valid_indices = [i for i, item in enumerate(batch_data["actions"]) if item]

            batch_data = {k: [v[i] for i in valid_indices] for k, v in batch_data.items()}

            # Now: make sure batch sizes match
            min_length = min(len(batch_data[k]) for k in batch_data.keys())
            batch_data = {k: v[:min_length] for k, v in batch_data.items()}


            processed_batch = {}
            num_unroll_steps = config.NUM_UNROLL_STEPS

            # Observations: simply stack (they are already proper tensors)
            processed_batch["observations"] = torch.stack(
                [obs.clone().detach() if isinstance(obs, torch.Tensor) else torch.tensor(obs) for obs in batch_data["observations"]]
            )

            # Sequences (actions, target_reward, target_value): pad or cut
            for key in ["actions", "target_reward", "target_value"]:
                items = batch_data[key]
                new_items = []
                for item in items:
                    if not item:  # If empty, skip it
                        continue

                    # Handle list of elements inside
                    if isinstance(item, list):
                        processed = []
                        for sub_item in item:
                            if isinstance(sub_item, Action):
                                processed.append(torch.tensor(sub_item.number))
                            else:
                                processed.append(torch.tensor(sub_item))
                        item = torch.stack(processed)
                    else:
                        if isinstance(item, Action):
                            item = torch.tensor(item.number)
                        else:
                            item = torch.tensor(item)

                    # Now item is a Tensor, fix its length
                    if item.dim() == 0:
                        item = item.unsqueeze(0)

                    if item.shape[0] < num_unroll_steps:
                        padding = torch.zeros(num_unroll_steps - item.shape[0], dtype=item.dtype)
                        item = torch.cat([item, padding])
                    elif item.shape[0] > num_unroll_steps:
                        item = item[:num_unroll_steps]

                    new_items.append(item)

                processed_batch[key] = torch.stack(new_items)



            # target_policy: assume it is already lists of dictionaries (visit counts)
            # Here we need to extract values properly
            policy_items = batch_data["target_policy"]
            new_policies = []
            for policy in policy_items:
                if isinstance(policy, list):
                    stacked = []
                    for p in policy:
                        policy_tensor = torch.zeros(config.ACTION_SPACE, dtype=torch.float32)
                        for action_idx, visit_count in p.items():
                            if action_idx < config.ACTION_SPACE:
                                policy_tensor[action_idx] = visit_count
                        stacked.append(policy_tensor)
                    policy_tensor = torch.stack(stacked)
                elif isinstance(policy, dict):
                    policy_tensor = torch.zeros(config.ACTION_SPACE, dtype=torch.float32)
                    for action_idx, visit_count in policy.items():
                        if action_idx < config.ACTION_SPACE:
                            policy_tensor[action_idx] = visit_count
                    policy_tensor = policy_tensor.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected type in policy_items: {type(policy)}")

                new_policies.append(policy_tensor)

            processed_batch["target_policy"] = torch.cat(new_policies, dim=0)

            batch_data = processed_batch



            loss = network_manager.train_step(batch_data)
            print(f"Batch {batch} loss: {loss:.4f}")
            all_losses.append(loss)

    """
    LETS FUCKING PLOT
    """
        # After training
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses, label="Training Loss")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()


