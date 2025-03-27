import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from project2.neural_net import MuZeroNetwork

class NeuralNetManager():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MuZeroNetwork(
            observation_dim=self.config.OBSERVATION_DIM,
            action_space=self.config.ACTION_SPACE,
            hidden_size=self.config.HIDDEN_SIZE,
            latent_dim=self.config.LATENT_DIM
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self._training_steps = 0
        
    def initial_inference(self, observation):
        observation = observation.to(self.device)
        return self.model.initial_inference(observation)
    
    def recurrent_inference(self, abstract_state, action):
        hidden_state = abstract_state.to(self.device)
        action = action.to(self.device)
        return self.model.recurrent_inference(hidden_state, action)

    def get_weights(self):
        """
        Returns the current network weights.
        """
        return self.model.get_weights()
    
    def get_training_steps(self):
        """
        Return the current trainig steps
        """
        return self.model.training_steps()
    
    def save(self, filepath):
        
        torch.save(self.model.state_dict(), filepath)
        
    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.model.increment_training_steps()
        return loss.item()

    
    def compute_loss(self, batch):
        observations = batch["observations"].to(self.device)           
        target_policies = batch["target_policy"]                        
        target_rewards = batch["target_reward"]                         
        target_values = batch["target_value"]                           
        actions = batch["actions"].to(self.device)                      

        
        out = self.model.initial_inference(observations)
        loss_policy = F.cross_entropy(out.policy_logits, target_policies[:, 0])
        loss_value = F.mse_loss(out.value, target_values[:, 0])
        total_reward_loss = 0.0

        hidden_state = out.hidden_state

        
        for k in range(actions.size(1)):
            out = self.model.recurrent_inference(hidden_state, actions[:, k])
            loss_policy += F.cross_entropy(out.policy_logits, target_policies[:, k + 1])
            loss_value += F.mse_loss(out.value, target_values[:, k + 1])
            total_reward_loss += F.mse_loss(out.reward, target_rewards[:, k])
            hidden_state = out.hidden_state  

        total_loss = loss_policy + loss_value + total_reward_loss
        return total_loss
