import torch
import numpy as np
import logging
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime

from project2 import config as default_config
from project2.neural_net_manager import NeuralNetManager
from project2.replay_buffer import ReplayBuffer
from project2.buffer_game import Game
from project2.playground_mcts_neural_model import play_and_record_game

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_muzero")

def train_muzero(config=default_config):
    num_iterations = config.NUM_ITERATIONS
    games_per_iteration = config.GAMES_PER_ITERATION
    training_steps_per_iteration = config.TRAINING_STEPS_PER_ITERATION
    checkpoint_frequency = config.CHECKPOINT_FREQUENCY
    log_dir = 'tensorboard_logs'
    os.makedirs("models", exist_ok=True)
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tb_log_dir = os.path.join(log_dir, f'muzero_{current_time}')
    
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logs will be saved to {tb_log_dir}")
    
    nnm = NeuralNetManager(config)
    replay_buffer = ReplayBuffer(config)
    
    total_games = 0
    total_steps = 0
    training_start_time = time.time()
    
    hparam_dict = {
        "lr": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "discount": config.DISCOUNT,
        "unroll_steps": config.NUM_UNROLL_STEPS,
        "latent_dim": config.LATENT_DIM,
        "hidden_size": config.HIDDEN_SIZE
    }
    writer.add_hparams(hparam_dict, {'status': 1})
    
    for iteration in tqdm(range(num_iterations), desc="Training iterations"):
        logger.info(f"Starting iteration {iteration+1}/{num_iterations}")
        iteration_start_time = time.time()
        
        logger.info(f"Self-play phase: Generating {games_per_iteration} games")
        game_rewards = []
        game_lengths = []
        
        for game_idx in tqdm(range(games_per_iteration), desc="Self-play games"):
            try:
                game = play_and_record_game(config, nnm)  
                
                
                total_reward = sum(game.rewards) if game.rewards else 0
                game_length = len(game.observations)
                   
                game_rewards.append(total_reward)
                game_lengths.append(game_length)
                    
                replay_buffer.save_game(game)
                total_games += 1
                    
            except Exception as e:
                logger.error(f"Error during self-play game {game_idx}: {e}")
        
        if game_rewards:
            writer.add_scalar('SelfPlay/AvgReward', np.mean(game_rewards), iteration)
            writer.add_scalar('SelfPlay/MaxReward', np.max(game_rewards), iteration)
            writer.add_scalar('SelfPlay/AvgGameLength', np.mean(game_lengths), iteration)
            writer.add_scalar('Data/BufferSize', len(replay_buffer.buffer), iteration)
        
        if len(replay_buffer.buffer) > 0:
            logger.info(f"Training phase: {training_steps_per_iteration} training steps")
            total_losses = []
            policy_losses = []
            value_losses = []
            reward_losses = []
            
            for step in tqdm(range(training_steps_per_iteration), desc="Training steps"):
                try:
                    batch = replay_buffer.sample_batch()
                    
                    loss_dict = nnm.train_step(batch)
                    
                    total_losses.append(loss_dict['total_loss'])
                    policy_losses.append(loss_dict['policy_loss'])
                    value_losses.append(loss_dict['value_loss'])
                    reward_losses.append(loss_dict['reward_loss'])
                except Exception as e:
                    logger.error(f"Error during training step: {e}")
            
            if total_losses:
                avg_total_loss = sum(total_losses) / len(total_losses)
                avg_policy_loss = sum(policy_losses) / len(policy_losses)
                avg_value_loss = sum(value_losses) / len(value_losses)
                avg_reward_loss = sum(reward_losses) / len(reward_losses)
                
                writer.add_scalar('Training/AvgTotalLoss', avg_total_loss, iteration)
                writer.add_scalar('Training/AvgPolicyLoss', avg_policy_loss, iteration)
                writer.add_scalar('Training/AvgValueLoss', avg_value_loss, iteration)
                writer.add_scalar('Training/AvgRewardLoss', avg_reward_loss, iteration)
                
                writer.add_scalars('Training/LossComponents', {
                    'Policy': avg_policy_loss,
                    'Value': avg_value_loss,
                    'Reward': avg_reward_loss
                }, iteration)
                
                logger.info(f"Iteration {iteration+1}: "
                           f"Total Loss: {avg_total_loss:.4f}, "
                           f"Policy: {avg_policy_loss:.4f}, "
                           f"Value: {avg_value_loss:.4f}, "
                           f"Reward: {avg_reward_loss:.4f}")
            else:
                logger.warning("No valid loss values for this iteration")
            
            for name, param in nnm.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, iteration)
        else:
            logger.warning("No games in buffer, skipping training phase")
        
        iteration_time = time.time() - iteration_start_time
        writer.add_scalar('Time/IterationTime', iteration_time, iteration)
        
        if (iteration + 1) % checkpoint_frequency == 0:
            model_path = f"models/muzero_checkpoint_{iteration+1}.pt"
            try:
                nnm.save(model_path)
                logger.info(f"Model checkpoint saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model checkpoint: {e}")
    
    nnm.save("models/muzero_final.pt")
    logger.info("Training complete! Final model saved.")
    
    total_time = time.time() - training_start_time
    writer.add_text('Summary', f'Total training time: {total_time:.2f}s, Games: {total_games}, Steps: {total_steps}')
    
    writer.close()
    
    return nnm, replay_buffer

if __name__ == "__main__":
    model, buffer = train_muzero(default_config)
    
