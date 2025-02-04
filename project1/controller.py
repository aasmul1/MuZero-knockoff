#traditional PID controller
'''Define the base controller class.
Subclass into:
Traditional PID controller: Update using the three parameters (kp, ki, kd).
AI-based controller: Neural network implementation using JAX'''

from jax import random
import jax
import jax.numpy as jnp

        

class ClassicPIDController():
    
    def __init__(self, learning_rate):
        self.params = {}
        self.learning_rate = learning_rate
        self.initialize_params()
        
    def initialize_params(self):
        self.params = {
            "kp": 0.1,
            "ki": 0.1,
            "kd": 0.1
        }
        return self.params.copy()  # Return a copy to avoid accidental modifications

        
    def get_errors(self):
        return self.errors
        
    def compute_control_signal(self, error, params):
        dEdt = error - self.errors[-1] if len(self.errors) > 0 else 0
        integral = jnp.sum(jnp.array(self.errors))
        return params["kp"] * error + params["ki"] * integral + params["kd"] * dEdt
    
    def update_error_history(self, error):
        self.errors.append(error)
            
    def update_params(self, grads):
        new_params = self.params.copy()  # Copy to avoid modifying original dict

        new_params["kp"] -= self.learning_rate * grads["kp"]
        new_params["ki"] -= self.learning_rate * grads["ki"]
        new_params["kd"] -= self.learning_rate * grads["kd"]

        return new_params  # Always return a dictionary

            
        
    def compute_mse(self):
        return jnp.mean(jnp.array(self.errors)**2)
    
    def reset(self):
        self.errors = []

class NeuralNetController():
    
    def __init__(self, hidden_layers, activation_layers, learning_rate):   
        self.hidden_layers = hidden_layers
        self.activation_layers = activation_layers
        self.learning_rate = learning_rate  
        self.params = {}
        self.initialize_params()
        self.initialize_activation_functions()
    
    def initialize_params(self):
        input_dim = 3
        keys = random.split(random.PRNGKey(0), len(self.hidden_layers) + 1)
        
        for i, layer_neurons in enumerate(self.hidden_layers):
            w_key, b_key = random.split(keys[i])
            self.params[f"W{i}"] = random.uniform(w_key, (input_dim, layer_neurons), minval=-0.1, maxval=0.1)
            self.params[f"b{i}"] = random.uniform(b_key, (layer_neurons,), minval=0, maxval=0.1)
            
            input_dim = layer_neurons
            
        w_key, b_key = random.split(keys[-1])
        self.params[f"W{len(self.hidden_layers)}"] = random.uniform(w_key, (input_dim, 1), minval=-0.1, maxval=0.1 )
        self.params[f"b{len(self.hidden_layers)}"] = random.uniform(b_key, (1,), minval=0, maxval=0.1)    
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
    
    def compute_control_signal(self, error, params):
        dEdt = error - self.error_history[-1] if len(self.error_history) > 0 else 0.0
        integral = jnp.sum(jnp.array(self.error_history))
        output = self.forward(jnp.array([[error, dEdt, integral]]), params)
        return output.squeeze()


    def update_params(self, grads, bias_range, weight_range):
        new_params = {}

        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

        num_layers = len(self.hidden_layers)
        
        weight_range = tuple(sorted(weight_range))
        bias_range = tuple(sorted(bias_range))

        
        for i in range(num_layers):
            updated_weight = self.params[f"W{i}"] - self.learning_rate * grads[f"W{i}"]
            new_params[f"W{i}"] = jnp.clip(updated_weight, weight_range[0], weight_range[1])
            
            # Oppdater biasverdiene og klipp til bias_range
            updated_bias = self.params[f"b{i}"] - self.learning_rate * grads[f"b{i}"]
            new_params[f"b{i}"] = jnp.clip(updated_bias, bias_range[0], bias_range[1])
        
        # For output-laget (forutsatt at output-laget følger etter de skjulte lagene)
        updated_weight = self.params[f"W{num_layers}"] - self.learning_rate * grads[f"W{num_layers}"]
        new_params[f"W{num_layers}"] = jnp.clip(updated_weight, weight_range[0], weight_range[1])
        
        updated_bias = self.params[f"b{num_layers}"] - self.learning_rate * grads[f"b{num_layers}"]
        new_params[f"b{num_layers}"] = jnp.clip(updated_bias, bias_range[0], bias_range[1])
        
        return new_params



            
