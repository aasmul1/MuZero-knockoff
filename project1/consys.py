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
from plant import BathtubModel, CournotModel, FuelTankModel
from config import consys_params, neural_network_params, bathtub_params, cournot_params, fuelTank_params

class ConSys():
    def __init__(self, params):
        self.target = 0
        self.params = params
        self.key = random.PRNGKey(0)
        
        self.controller = self.initialize_controller()
        self.plant = self.initialize_plant()
        
    def initialize_controller(self):
        """Velger riktig controller basert på config."""
        if self.params["controller"] == "NeuralNetController":
            return NeuralNetController(
                hidden_layers=neural_network_params["hidden_layers"], 
                activation_layers=neural_network_params["activation_layers"], 
                learning_rate=consys_params["learning_rate"]
            )
        elif self.params["controller"] == "ClassicPIDController":
            return ClassicPIDController(consys_params["learning_rate"])
        else:
            raise ValueError(f"Ukjent controller: {self.params['controller']}")

    def initialize_plant(self):
        """Velger riktig plant basert på config."""
        if self.params["plant"] == "Bathtub":
            self.target = bathtub_params["target"]
            return BathtubModel(
                bathtub_params["area_bathtub"], 
                bathtub_params["area_drain"], 
                bathtub_params["water_height"]
            )
        elif self.params["plant"] == "Cournot":
            self.target = cournot_params["target"]
            return CournotModel(
                cournot_params["p_max"], 
                cournot_params["marginal_cost"]
            )
        elif self.params["plant"] == "FuelTank":
            self.target = fuelTank_params["target"]
            return FuelTankModel(
                fuelTank_params["consumption_rate"], 
                fuelTank_params["max_cap"], 
                fuelTank_params["initial_fuel"]
            )
        else:
            raise ValueError(f"Ukjent plant: {self.params['plant']}")
        
    def get_disturbance(self):
        """Henter en tilfeldig forstyrrelse."""
        min_val, max_val = -0.01, 0.01
        self.key, subkey = random.split(self.key)
        return random.uniform(subkey, minval=min_val, maxval=max_val)

    def run_system(self):
        def mse_fn(params):
            return self.run_one_epoch(params)

        self.controller.reset()
        
        # Ensure ClassicPIDController initializes correctly
        if isinstance(self.controller, ClassicPIDController):
            params = self.controller.initialize_params()  # Returns a dictionary
        else:
            params = self.controller.initialize_params()  # NeuralNetController params

        gradfunc = jax.value_and_grad(mse_fn)
        errors = []
        params_history = []

        for _ in range(consys_params["epochs"]):
            avg_mse, grads = gradfunc(params)
            errors.append(avg_mse)
            params_history.append(params)

            if isinstance(self.controller, NeuralNetController):
                params = self.controller.update_params(grads, 
                                                    bias_range=neural_network_params["bias_range"], 
                                                    weight_range=neural_network_params["weight_range"])
            else:  # Classic PID
                params = self.controller.update_params(grads)

            self.controller.params = params  # Ensure params are stored properly

        return errors, params_history

    
    def run_one_epoch(self, params):
        """Kjører én simuleringsepoke."""
        self.controller.reset()
        control_signal = 0.0
        plant = self.plant.deep_copy()
                
        timestep = consys_params["timesteps"]
        disturbance = jnp.array([self.get_disturbance() for _ in range(timestep)])
        
        for i in range(timestep):
            output = plant.calculate_output(control_signal, disturbance[i])
            error = self.target - output
            control_signal = self.controller.compute_control_signal(error, params)  
            self.controller.update_error_history(error)
            
        return self.controller.compute_mse()

    def plot_results(self, mse, params_history):
        """Plot MSE and parameter evolution over epochs side by side."""
        plt.figure(figsize=(14, 6))  # Wider figure for side-by-side plots

        # First plot: MSE vs Epochs
        plt.subplot(1, 2, 1)  # Change to (1, 2, X) for side-by-side
        plt.plot(mse, label="MSE")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title("MSE vs Epochs")
        plt.legend()
        plt.grid()

        # Second plot: PID Parameters vs Epochs (only for ClassicPIDController)
        if isinstance(self.controller, ClassicPIDController):
            params_array = jnp.array([[p["kp"], p["ki"], p["kd"]] for p in params_history])

            plt.subplot(1, 2, 2)  # Position it beside the first plot
            plt.plot(params_array[:, 0], label="kp")
            plt.plot(params_array[:, 1], label="ki")
            plt.plot(params_array[:, 2], label="kd")
            plt.xlabel("Epochs")
            plt.ylabel("Parameter Values")
            plt.title("PID Parameters vs Epochs")
            plt.legend()
            plt.grid()

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()


if __name__ == "__main__":
    consys = ConSys(consys_params)
    mse, params_history = consys.run_system()
    consys.plot_results(mse, params_history)


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
        elif self.params["plant"] == "Cournot":
            return CournotModel(5, 0.1)
        elif self.params["plant"] == "FuelTank":
            return FuelTankModel(0.01, 10, 5)
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
        learning_rate = 0.01
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
        target = 7
        
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
        elif self.params["plant"] == "Cournot":
            return CournotModel(5, 0.1)
        elif self.params["plant"] == "FuelTank":
            return FuelTankModel(0.1, 10, 5)
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


        for _ in range(100):

            avg_mse, grads = gradfunc(params)
            params = self.controller.update_params(grads)


            errors.append(avg_mse)
            
            params_history.append(params)


            # Update parameters using gradient descent
            params = self.controller.update_params(grads)
            self.controller.params = params  # Store back in the controller


        return errors, params_history
    
    
    def run_one_epoch(self, params):
        self.controller.reset()
        control_signal = 0.0
        target = 6
        
        plant = self.plant.deep_copy()
                
        timestep = 25
        disturbance = jnp.array([self.get_disturbance() for _ in range(timestep)])
        
        for i in range(timestep):
            output = plant.calculate_output(control_signal, disturbance[i])
            error = target - output
            control_signal = self.controller.compute_control_signal(error, params)  
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
        plt.show()
        

        

        
        
