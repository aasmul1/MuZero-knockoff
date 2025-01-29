#traditional PID controller
'''Define the base controller class.
Subclass into:
Traditional PID controller: Update using the three parameters (kp, ki, kd).
AI-based controller: Neural network implementation using JAX'''

from jax import random
import jax.numpy as jnp

        

class ClassicPIDController():
    
    def __init__(self):
        self.initialize_params(0.1, 0.1, 0.1)
        
    def initialize_params(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
    def get_errors(self):
        return self.errors
        
    def compute_control_signal(self, error):
        dEdt = error - self.errors[-1] if len(self.errors) > 0 else 0
        integral = jnp.sum(jnp.array(self.errors))
        return self.kp * error + self.ki * integral + self.kd * dEdt
    
    def update_error(self, error):
        self.errors.append(error)
        
    def compute_mse(self):
        return jnp.mean(jnp.array(self.errors)**2)
    
    def reset(self):
        self.errors = []

class NeuralNetController():
    
    def __init__(self):
        self.hidden_layers = [4, 2]
        self.activation_layers = ["relu", "relu"]
        self.params = {}
        self.initialize_params()
        self.initialize_activation_functions()
    
    def initialize_params(self):
        input_dim = 3
        keys = random.split(random.PRNGKey(0), len(self.hidden_layers) + 1)
        
        for i, layer_neurons in enumerate(self.hidden_layers):
            w_key, b_key = random.split(keys[i])
            self.params[f"W{i}"] = random.uniform(w_key, (input_dim, layer_neurons))
            self.params[f"b{i}"] = random.uniform(b_key, (layer_neurons,))
            
            input_dim = layer_neurons
            
        w_key, b_key = random.split(keys[-1])
        self.params[f"W{len(self.hidden_layers)}"] = random.uniform(w_key, (input_dim, 1))
        self.params[f"b{len(self.hidden_layers)}"] = random.uniform(b_key, (1,))    
        return self.params    
            
    
    def initialize_activation_functions(self):
        activation_functions = []
        for i in self.activation_layers:
            if i == 'relu':
                activation_functions.append(self.relu)
            elif i == 'sigmoid':
                activation_functions.append(self.sigmoid)
            elif i == 'tanh':
                activation_functions.append(self.tanh)
            else:
                raise ValueError("Activation function not supported")
        self.activation_functions = activation_functions
        
    def update_error_history(self, error):
        self.error_history.append(error)
        
    def reset(self):
        self.error_history = []
    
    def compute_mse(self):
        return jnp.mean(jnp.array(self.error_history)**2)
                
    def relu(self, x):
        return jnp.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))
    
    def tanh(self, x):
        return jnp.tanh(x)
    
    def forward(self, x, params):
        for i in range(len(self.hidden_layers)):
            weights = params[f"W{i}"]
            biases = params[f"b{i}"]
            x = jnp.dot(x, weights) + biases
            x = self.activation_functions[i](x)
        weights = params[f"W{len(self.hidden_layers)}"]
        biases = params[f"b{len(self.hidden_layers)}"]
        x = jnp.dot(x, weights) + biases
        return x
    
    def compute_control_signal(self, error):
        dEdt = error - self.error_history[-1] if len(self.error_history) > 0 else 0.0
        integral = jnp.sum(jnp.array(self.error_history))
        output = self.forward(jnp.array([error, dEdt, integral]), self.params)
        print("Output:", output)
        return output
    
    def update_params(self, params):
        for i in range(len(self.hidden_layers)):
            weights = params[f"W{i}"]
            biases = params[f"b{i}"]
            self.params[f"W{i}"] -= 0.01 * weights
            self.params[f"b{i}"] -= 0.01 * biases
        weights = params[f"W{len(self.hidden_layers)}"]
        self.params[f"W{len(self.hidden_layers)}"] -= 0.01 * weights
        biases = params[f"b{len(self.hidden_layers)}"]
        self.params[f"b{len(self.hidden_layers)}"] -= 0.01 * biases
        return params
            
