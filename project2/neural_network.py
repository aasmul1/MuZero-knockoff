import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple container for network outputs.
class NetworkOutput:
    def __init__(self, value, reward, policy_logits, hidden_state):
        self.value = value            
        self.reward = reward          
        self.policy_logits = policy_logits  
        self.hidden_state = hidden_state    

class MuZeroNetwork(nn.Module):
    def __init__(self, observation_dim, action_space, hidden_size, latent_dim):
        """
        Args:
            observation_dim (int): Dimension of the flattened observation.
            action_space_size (int): Number of available actions.
            hidden_size (int): Number of neurons in hidden layers.
            latent_dim (int): Size of the latent (abstract) state.
        """
        super(MuZeroNetwork, self).__init__()
        self.action_space_size = len(action_space)
        self.latent_dim = latent_dim
        
        self.representation_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim)
            
        )
        
        self.prediction_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_space_size + 1)
        )
        
        self.dynamics_net = nn.Sequential(
            nn.Linear(latent_dim + self.action_space_size , hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim + 1)
            
        )
        
    def represent_state(self, observation):
        
        return self.representation_net(observation)
    
    def predict(self, abstract_state):
        output = self.prediction_net(abstract_state)
        policy_logits = output[:, :self.action_space_size]
        value = output[:, self.action_space_size]

        return policy_logits, value
    
    def transition(self, abstract_state, action):
        action_one_hot = F.one_hot(action, num_classes=self.action_space_size).float()
        x = torch.cat([abstract_state, action_one_hot], dim=-1)
        dynamics_out = self.dynamics_net(x)
        next_state = dynamics_out[:, :self.latent_dim]
        reward = dynamics_out[:, self.latent_dim]
        return next_state, reward


# if __name__ == "__main__":
#     # Hyperparameters
#     observation_dim = 100      # e.g., flattened 10x10 board => 100-dimensional vector.
#     action_space = [0, 1, 2, 3, 4]  # For example, 5 possible discrete actions.
#     hidden_size = 128          # Size of hidden layers.
#     latent_dim = 64            # Latent state dimension.
    
#     # Instantiate the network.
#     net = MuZeroNetwork(observation_dim, action_space, hidden_size, latent_dim)
    
#     # Create a dummy observation.
#     dummy_obs = torch.randn(1, observation_dim)
    
#     # Perform initial inference: representation + prediction.
#     latent = net.represent_state(dummy_obs)
#     policy_logits, value = net.predict(latent)
#     initial_output = NetworkOutput(value, torch.zeros_like(value), policy_logits, latent)
    
#     print("Initial Inference:")
#     print("Value:", initial_output.value)
#     print("Policy Logits:", initial_output.policy_logits)
#     print("Latent State:", initial_output.hidden_state)
    
#     # Perform recurrent inference: dynamics + prediction.
#     # Suppose we take a dummy action (e.g., action index 2).
#     dummy_action = torch.tensor([2])
#     next_latent, reward = net.transition(latent, dummy_action)
#     next_policy_logits, next_value = net.predict(next_latent)
#     recurrent_output = NetworkOutput(next_value, reward, next_policy_logits, next_latent)
    
#     print("\nRecurrent Inference:")
#     print("Value:", recurrent_output.value)
#     print("Reward:", recurrent_output.reward)
#     print("Policy Logits:", recurrent_output.policy_logits)
#     print("New Latent State:", recurrent_output.hidden_state)