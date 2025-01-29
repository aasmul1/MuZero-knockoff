import jax
from jax import random, tree_map
import matplotlib.pyplot as plt
import tqdm as tqdm
import jax.numpy as jnp

'''Manage the overall system.
Initialize the plant and controller.
Run simulations for epochs and timesteps.
Handle configurations and logging.'''

from controller import ClassicPIDController, NeuralNetController
from plant import BathtubModel


class ConSysClassic():
    def __init__(self, params):
        self.params = params
        self.controller = self.initialize_controller()
        self.plant = self.initialize_plant()
        self.key = random.PRNGKey(0)
        
    def initialize_controller(self):
        
        if self.params["controller"] == "ClassicPID":
            return ClassicPIDController()
        else:
            raise ValueError("Controller not supported")
        
    def initialize_plant(self):
        
        if self.params["plant"] == "Bathtub":
            return BathtubModel(3.0, 0.01, 15)
        else:
            raise ValueError("Plant not supported")
        
    def get_disturbance(self):
        min_val = -0.01
        max_val = 0.01
        self.key, subkey = random.split(self.key)

        return random.uniform(
            key=subkey, minval=min_val, maxval=max_val
        )    
        
    def run_system(self):
        def mse_fn(params):
            """Wrapper function for MSE calculation with given parameters."""
            self.controller.kp, self.controller.ki, self.controller.kd = params
            return self.run_one_epoch()

        # Initialize parameters
        params = jnp.array([self.controller.kp, self.controller.ki, self.controller.kd])
        # Prepare gradient function
        gradfunc = jax.value_and_grad(mse_fn)

        errors = []
        learning_rate = 0.001
        params_history = []


        for _ in range(40):
            avg_mse, grads = gradfunc(params)

            errors.append(avg_mse)
            
            params_history.append(params)


            # Update parameters using gradient descent
            params = params - learning_rate * grads
            print(grads)

            # Store updated parameters
            self.controller.kp, self.controller.ki, self.controller.kd = params

        return errors, params_history
    
    
    def run_one_epoch(self):
        self.controller.reset()
        control_signal = 0.0
        target = 15.0
        
        plant = self.plant.deep_copy()
                
        timestep = 25
        disturbance = jnp.array([self.get_disturbance() for _ in range(timestep)])
        
        for i in range(timestep):
            output = plant.calculate_output(control_signal, disturbance[i])
            error = target - output
            control_signal = self.controller.compute_control_signal(error)  
            self.controller.update_error(error)
            
        mse = self.controller.compute_mse()
        
        return mse
    
    def plot_results(self, mse, params_history):
        # Plot MSE vs Epochs
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(mse, label="MSE")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title("MSE vs Epochs")
        plt.legend()
        plt.grid()

        # Plot PID parameters vs Epochs
        params_array = jnp.array(params_history)  # Convert to JAX array for indexing
        plt.subplot(2, 1, 2)
        plt.plot(params_array[:, 0], label="kp")
        plt.plot(params_array[:, 1], label="ki")
        plt.plot(params_array[:, 2], label="kd")
        plt.xlabel("Epochs")
        plt.ylabel("Parameter Values")
        plt.title("PID Parameters vs Epochs")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

class ConSysNeural():
    def __init__(self, params):
        self.params = params
        self.controller = self.initialize_controller()
        self.plant = self.initialize_plant()
        self.key = random.PRNGKey(0)
        
    def initialize_controller(self):
        
        if self.params["controller"] == "NeuralNet":
            return NeuralNetController()
        else:
            raise ValueError("Controller not supported")
        
    def initialize_plant(self):
        
        if self.params["plant"] == "Bathtub":
            return BathtubModel(1.0, 0.01, 15)
        else:
            raise ValueError("Plant not supported")
        
    def get_disturbance(self):
        min_val = -0.01
        max_val = 0.01
        self.key, subkey = random.split(self.key)

        return random.uniform(
            key=subkey, minval=min_val, maxval=max_val
        )    
        
    def run_system(self):
        def mse_fn(params):
            """Wrapper function for MSE calculation with given parameters."""
            return self.run_one_epoch(params)

        self.controller.reset()
        self.controller.initialize_activation_functions()
        params = self.controller.initialize_params()
        # Prepare gradient function
        
        gradfunc = jax.value_and_grad(mse_fn)

        errors = []
        params_history = []


        for _ in range(40):
            avg_mse, grads = gradfunc(params)

            errors.append(avg_mse)
            
            params_history.append(params)


            # Update parameters using gradient descent
            params = self.controller.update_params(grads)


        return errors, params_history
    
    
    def run_one_epoch(self, params):
        self.controller.reset()
        control_signal = 0.0
        target = 15.0
        
        plant = self.plant.deep_copy()
                
        timestep = 25
        disturbance = jnp.array([self.get_disturbance() for _ in range(timestep)])
        
        for i in range(timestep):
            output = plant.calculate_output(control_signal, disturbance[i])
            error = target - output
            control_signal = self.controller.compute_control_signal(error)  
            self.controller.update_error_history(error)
            
        mse = self.controller.compute_mse()
        
        return mse

    
    def plot_results(self, mse):
        # Plot MSE vs Epochs
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(mse, label="MSE")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title("MSE vs Epochs")
        plt.legend()
        plt.grid()

        

        
        
if __name__ == "__main__":
    params = {
        "plant": "Bathtub",
        "controller": "NeuralNet",
    }
    consys = ConSysNeural(params)
    mse, params_history = consys.run_system()  # Capture both MSE and parameter history
    consys.plot_results(mse, params_history)