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
   
        self.hidden_layers = [3, 2]
        self.activation_layers = ["tanh", "tanh"]
        self.params = {}
        self.initialize_params({"min": -0.1, "max": 0.1}, {"min": 0, "max": 0.1})
        self.initialize_activation_layers()
        
    def initialize_params(self, weights_range, biases_range):
        
        keys = random.split(random.PRNGKey(0), len(self.hidden_layers) + 1)
        input_dim = 3
        
        for i, layer_neurons in enumerate(self.hidden_layers):
            w_key, b_key = random.split(keys[i])
            self.params[f'layer_{i}'] = {
            "weights": random.uniform(w_key, (input_dim, layer_neurons), minval=weights_range["min"], maxval=weights_range["max"]),
            "biases": random.uniform(b_key, (layer_neurons,), minval=biases_range["min"], maxval=biases_range["max"]),
        }
            input_dim = layer_neurons
            
        w_key, b_key = random.split(keys[-1])
        self.params[f'layer_{len(self.hidden_layers)}'] = {
        "weights": random.uniform(w_key, (input_dim, 1), minval=weights_range["min"], maxval=weights_range["max"]),
        "biases": random.uniform(b_key, (1,), minval=biases_range["min"], maxval=biases_range["max"]),
    }
        
    
        
    def initialize_activation_layers(self):
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
        
    def compute_mse(self):
        return jnp.mean(jnp.array(self.error_history)**2)
    
    def get_params(self):
        
        return self.params
                
    def relu(self, x):
        return jnp.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))
    
    def tanh(self, x):
        return jnp.tanh(x)
    
    def dot(self, x, w, b):
        return jnp.dot(x, w) + b
    
    def forward(self, x, params):
        # 1) Pass x through each hidden layer in turn
        for i in range(len(self.hidden_layers)):
            layer = params[f'layer_{i}']
            x = self.activation_functions[i](self.dot(x, layer["weights"], layer["biases"]))

        # 2) After the loop, pass x through the final (output) layer exactly once
        out_layer = params[f'layer_{len(self.hidden_layers)}']
        x = self.dot(x, out_layer["weights"], out_layer["biases"])
        print("Forward Output:", x[0])
        return x[0]
    
    def compute_control_signal(self, error, params):
        dEdt = error - self.error_history[-1] if len(self.error_history) > 0 else 0.0
        integral = jnp.sum(jnp.array(self.error_history))
        nn_input = jnp.array([error, dEdt, integral])
        print("NN Input:", nn_input)
        nn_output = self.forward(nn_input, params)
        print("NN Output:", nn_output)
        return nn_output
    
    def reset(self):
        """
        Resets the controller's error history.
        """
        self.error_history = []
        
    def update_error_history(self, error):
        
        self.error_history.append(error)
        
    def update_params(self, grads):
    # For each layer
        for layer_key in self.params.keys():
            # self.params[layer_key] is also a dict: {"weights": ..., "biases": ...}
            # grads[layer_key] is also a dict: {"weights": ..., "biases": ...}

            self.params[layer_key]["weights"] -= 0.01 * grads[layer_key]["weights"]
            self.params[layer_key]["biases"]  -= 0.01 * grads[layer_key]["biases"]


