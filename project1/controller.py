from jax import random
import jax
import jax.numpy as jnp

class ClassicPIDController():
    """
    A simple, discrete-time PID controller with a gradient-based parameter update rule.

    Attributes:
        learning_rate (float): Controls the speed of gradient-based parameter updates.
        params (dict): Current dictionary of PID parameters { "kp", "ki", "kd" }.
        errors (list): History of recent errors to compute derivative and integral terms.
    """
    
    def __init__(self, learning_rate):
        """
        Initialize the PID controller with a given learning rate, 
        set default PID gains, and reset the error history.

        Args:
            learning_rate (float): Step size for parameter updates.
        """
        self.params = {}
        self.learning_rate = learning_rate
        self.initialize_params()
        self.reset()
        
    def initialize_params(self):
        """
        Initialize PID parameters to default values.

        Returns:
            dict: A copy of the initialized { "kp", "ki", "kd" } parameters.
        """
        self.params = {
            "kp": 0.1,
            "ki": 0.1,
            "kd": 0.1
        }
        return self.params.copy()
        
    def get_errors(self):
        """
        Provide the current error history.

        Returns:
            list: The list of recent errors encountered by the controller.
        """
        return self.errors
        
    def compute_control_signal(self, error, params):
        """
        Compute the PID output using the provided PID gains and the current error.

        Args:
            error (float): The current error (target - output).
            params (dict): The PID gains { "kp", "ki", "kd" } to use.

        Returns:
            float: The control signal to be applied to the plant.
        """
        dEdt = error - self.errors[-1] if len(self.errors) > 0 else 0
        integral = jnp.sum(jnp.array(self.errors))
        return params["kp"] * error + params["ki"] * integral + params["kd"] * dEdt
    
    def update_error_history(self, error):
        """
        Add the current error to the error history list.

        Args:
            error (float): The current error (target - output).
        """
        self.errors.append(error)
            
    def update_params(self, grads):
        """
        Update the PID parameters using gradient descent.

        Args:
            grads (dict): The gradient of the loss w.r.t. each PID parameter.

        Returns:
            dict: A new dictionary of updated PID parameters.
        """
        new_params = self.params.copy()
        new_params["kp"] -= self.learning_rate * grads["kp"]
        new_params["ki"] -= self.learning_rate * grads["ki"]
        new_params["kd"] -= self.learning_rate * grads["kd"]

        return new_params

    def compute_mse(self):
        """
        Compute the mean squared error over the current error history.

        Returns:
            float: The average of error^2 for the stored error history.
        """
        return jnp.mean(jnp.array(self.errors)**2)
    
    def reset(self):
        """
        Reset the controller's error history.
        """
        self.errors = []


class NeuralNetController():
    """
    A simple neural network-based controller. Uses a feedforward NN to compute the control signal
    given the current error, its derivative, and the integral of errors.

    Attributes:
        hidden_layers (list of int): Number of units in each hidden layer.
        activation_layers (list of str): Activation function names for each hidden layer.
        learning_rate (float): Gradient descent step size for parameter updates.
        params (dict): Weights and biases for each layer of the network.
        activation_functions (list of callable): The actual function objects for each activation.
        error_history (list): A record of past errors to compute derivative and integral terms.
    """

    def __init__(self, hidden_layers, activation_layers, learning_rate, weight_range, bias_range):
        """
        Initialize the neural network controller with given architecture and learning rate,
        then create default parameter values and activation functions.

        Args:
            hidden_layers (list): List of neuron counts for each hidden layer.
            activation_layers (list): List of activation function names (e.g. "relu", "sigmoid", "tanh").
            learning_rate (float): Step size for parameter updates.
        """
        self.hidden_layers = hidden_layers
        self.activation_layers = activation_layers
        self.learning_rate = learning_rate
        self.params = {}
        
        self.initialize_params(weight_range, bias_range)
        self.initialize_activation_functions()
        self.reset()
    
    def initialize_params(self, weight_range, bias_range):
        """
        Create weight and bias matrices for each layer (including the output layer),
        storing them in self.params.

        Returns:
            dict: A dictionary containing "W0", "b0", ..., "Wn", "bn" for n layers.
        """
        input_dim = 3  
        keys = random.split(random.PRNGKey(0), len(self.hidden_layers) + 1)
        weight_range = tuple(sorted(weight_range))
        bias_range = tuple(sorted(bias_range))
        
        for i, layer_neurons in enumerate(self.hidden_layers):
            w_key, b_key = random.split(keys[i])
            self.params[f"W{i}"] = random.uniform(w_key, (input_dim, layer_neurons), minval=weight_range[0], maxval=weight_range[1])
            self.params[f"b{i}"] = random.uniform(b_key, (layer_neurons,), minval=bias_range[0], maxval=bias_range[1])
            input_dim = layer_neurons
            
        w_key, b_key = random.split(keys[-1])
        self.params[f"W{len(self.hidden_layers)}"] = random.uniform(
            w_key, (input_dim, 1), minval=-0.1, maxval=0.1
        )
        self.params[f"b{len(self.hidden_layers)}"] = random.uniform(
            b_key, (1,), minval=0, maxval=0.1
        )
        return self.params
    
    def initialize_activation_functions(self):
        """
        Translate the string identifiers in self.activation_layers into actual functions
        and store them in self.activation_functions.
        """
        activation_functions = []
        for act in self.activation_layers:
            if act == 'relu':
                activation_functions.append(self.relu)
            elif act == 'sigmoid':
                activation_functions.append(self.sigmoid)
            elif act == 'tanh':
                activation_functions.append(self.tanh)
            else:
                raise ValueError("Activation function not supported")
        self.activation_functions = activation_functions
        
    def update_error_history(self, error):
        """
        Add the current error to the history for derivative and integral calculations.

        Args:
            error (float): The current control error (target - output).
        """
        self.error_history.append(error)
        
    def reset(self):
        """
        Clear the error history.
        """
        self.error_history = []
    
    def compute_mse(self):
        """
        Compute the mean squared error over the stored error history.

        Returns:
            float: Mean of error^2 for all recorded errors.
        """
        return jnp.mean(jnp.array(self.error_history)**2)
    
    def relu(self, x):
        """
        ReLU activation: max(0, x).

        Args:
            x (array-like): Input values.

        Returns:
            float: The output of the relu function.
        """
        return jnp.maximum(0, x)
    
    def sigmoid(self, x):
        """
        Sigmoid activation: 1 / (1 + e^(-x)).

        Args:
            x (array-like): Input values.

        Returns:
            float: The output of the sigmoid function.
        """
        return 1 / (1 + jnp.exp(-x))
    
    def tanh(self, x):
        """
        Tanh activation: (e^x - e^-x) / (e^x + e^-x).

        Args:
            x (array-like): Input values.

        Returns:
            float: The output of the tanh function.
        """
        return jnp.tanh(x)
    
    def forward(self, x, params):
        """
        Forward pass through the neural net. For each hidden layer i:
          x = activation(Wi * x + bi).
        Then for the output layer:
          x = W_final * x + b_final.

        Args:
            x (array-like): Input of shape (batch_size, input_dim).
            params (dict): Dictionary of all layer weights and biases.

        Returns:
            array-like: The model's output, shape (batch_size, 1).
        """
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
        """
        Compute the control signal by feeding [error, derivative, integral] through the NN.

        Args:
            error (float): The current control error (target - output).
            params (dict): Weights and biases of the neural net.

        Returns:
            float: A scalar control output.
        """
        dEdt = error - self.error_history[-1] if len(self.error_history) > 0 else 0.0
        integral = jnp.sum(jnp.array(self.error_history))
        x_input = jnp.array([[error, dEdt, integral]])
        output = self.forward(x_input, params)
        return output.squeeze()
    
    def update_params(self, grads, bias_range, weight_range):
        """
        Update network parameters using gradient descent, clipping to given ranges.

        The gradient is clipped to [-1, 1] to avoid exploding gradients.
        Then weights and biases are updated and clipped to [weight_range, bias_range] respectively.

        Args:
            grads (dict): A dict of gradients with the same keys as self.params.
            bias_range (tuple): (min_bias, max_bias).
            weight_range (tuple): (min_weight, max_weight).

        Returns:
            dict: A new parameter dictionary with updated weights and biases.
        """
        new_params = {}
        
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        num_layers = len(self.hidden_layers)
        
        weight_range = tuple(sorted(weight_range))
        bias_range = tuple(sorted(bias_range))

        for i in range(num_layers):
            updated_weight = self.params[f"W{i}"] - self.learning_rate * grads[f"W{i}"]
            new_params[f"W{i}"] = jnp.clip(updated_weight, weight_range[0], weight_range[1])
            
            updated_bias = self.params[f"b{i}"] - self.learning_rate * grads[f"b{i}"]
            new_params[f"b{i}"] = jnp.clip(updated_bias, bias_range[0], bias_range[1])
        
        updated_weight = self.params[f"W{num_layers}"] - self.learning_rate * grads[f"W{num_layers}"]
        new_params[f"W{num_layers}"] = jnp.clip(updated_weight, weight_range[0], weight_range[1])
        
        updated_bias = self.params[f"b{num_layers}"] - self.learning_rate * grads[f"b{num_layers}"]
        new_params[f"b{num_layers}"] = jnp.clip(updated_bias, bias_range[0], bias_range[1])
        
        return new_params
