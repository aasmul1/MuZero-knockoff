import jax
from jax import random, tree_map
import matplotlib.pyplot as plt
import tqdm as tqdm
import jax.numpy as jnp

# Controllers and plants imported from other modules
from controller import ClassicPIDController, NeuralNetController
from plant import BathtubModel, CournotModel, FuelTankModel
from config import consys_params, neural_network_params, bathtub_params, cournot_params, fuelTank_params

class ConSys():
    """
    A generic control system class that sets up and runs simulations using
    a chosen controller (PID or neural net) and a chosen plant (Bathtub, Cournot, or FuelTank).

    Attributes:
        target (float): The control target (setpoint) for the plant's output.
        params (dict): Configuration parameters that determine which controller/plant to use.
        key (PRNGKey): JAX random key for generating random disturbances.
        controller (object): An instance of either ClassicPIDController or NeuralNetController.
        plant (object): An instance of one of the plant classes (BathtubModel, CournotModel, FuelTankModel).
    """
    def __init__(self, params):
        """
        Initialize the ConSys with given parameters, create the chosen controller and plant.

        Args:
            params (dict): Contains keys such as 'controller' and 'plant' to decide which
                           controller and plant to instantiate.
        """
        self.target = 0
        self.params = params
        self.key = random.PRNGKey(0)
        
        self.controller = self.initialize_controller()

        self.plant = self.initialize_plant()
        
    def initialize_controller(self):
        """
        Choose and initialize the correct controller based on self.params["controller"].

        Returns:
            object: An instance of either NeuralNetController or ClassicPIDController.
        """
        if self.params["controller"] == "NeuralNetController":
            return NeuralNetController(
                hidden_layers=neural_network_params["hidden_layers"], 
                activation_layers=neural_network_params["activation_layers"], 
                learning_rate=consys_params["learning_rate"],
                weight_range=neural_network_params["weight_range"],
                bias_range=neural_network_params["bias_range"]
            )
        elif self.params["controller"] == "ClassicPIDController":
            return ClassicPIDController(consys_params["learning_rate"])
        else:
            raise ValueError(f"Ukjent controller: {self.params['controller']}")

    def initialize_plant(self):
        """
        Choose and initialize the correct plant based on self.params["plant"].

        Returns:
            object: An instance of BathtubModel, CournotModel, or FuelTankModel.
        """
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
        """
        Generate a small random disturbance in the range [-0.01, 0.01].

        Returns:
            float: A random disturbance value.
        """
        min_val, max_val = -0.01, 0.01
        self.key, subkey = random.split(self.key)
        return random.uniform(subkey, minval=min_val, maxval=max_val)

    def run_system(self):
        """
        Main loop for training the controller to minimize MSE over multiple epochs.

        1. Resets the controller.
        2. Initializes parameters (PID gains or neural net weights).
        3. Defines a function `mse_fn` that runs one training epoch and returns the MSE.
        4. Uses JAX's `value_and_grad` to compute gradients of MSE w.r.t. parameters.
        5. Updates parameters and tracks the MSE evolution over epochs.

        Returns:
            (list, list):
                - A list of MSE values per epoch.
                - A list of parameter sets per epoch.
        """
        def mse_fn(params):
            return self.run_one_epoch(params)

        self.controller.reset()
        
        if isinstance(self.controller, ClassicPIDController):
            params = self.controller.initialize_params()
        else:
            params = self.controller.initialize_params(neural_network_params["weight_range"], neural_network_params["bias_range"])

        gradfunc = jax.value_and_grad(mse_fn)
        errors = []
        params_history = []

        
        for _ in range(consys_params["epochs"]):
            avg_mse, grads = gradfunc(params)
            errors.append(avg_mse)
            params_history.append(params)

            if isinstance(self.controller, NeuralNetController):
                params = self.controller.update_params(
                    grads,
                    bias_range=neural_network_params["bias_range"],
                    weight_range=neural_network_params["weight_range"]
                )
            else:
                params = self.controller.update_params(grads)

            self.controller.params = params

        return errors, params_history

    def run_one_epoch(self, params):
        """
        Run one simulation epoch (one pass over the system), using 'params' as the current
        controller parameters.

        1. Resets the controller state.
        2. Creates a deep copy of the plant's initial state.
        3. Over a fixed number of timesteps, applies a control signal and disturbance,
           updates the plant, and calculates the error.
        4. Accumulates error in the controller to compute MSE.

        Args:
            params: The current parameter set for the controller.

        Returns:
            float: The Mean Squared Error for this epoch.
        """
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
        """
        Plot the MSE curve over epochs, and (if using PID) the evolution of PID parameters over epochs.

        Args:
            mse (list): A list of MSE values, one per epoch.
            params_history (list): A list of parameter states, one per epoch.
        """
        plt.figure(figsize=(14, 6))

        # Plot MSE vs. epochs
        plt.subplot(1, 2, 1)
        plt.plot(mse, label="MSE")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title("MSE vs Epochs")
        plt.legend()
        plt.grid()

        if isinstance(self.controller, ClassicPIDController):
            params_array = jnp.array([[p["kp"], p["ki"], p["kd"]] for p in params_history])

            plt.subplot(1, 2, 2)
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


if __name__ == "__main__":
    consys = ConSys(consys_params)
    mse, params_history = consys.run_system()
    consys.plot_results(mse, params_history)

