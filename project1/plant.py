import jax.numpy as jnp
from jax import random

class BathtubModel:
    """
    A discrete-time simulation of water height in a bathtub model.
    
    Attributes:
        area_bathtub (float): Cross-sectional area of the bathtub.
        area_drain (float): Cross-sectional area of the drain.
        water_height (float): Current water height in the bathtub.
        intial_water_height (float): The stored initial water height (for reset).
        velocity (float): Current water outflow velocity from the bathtub.
        flow_rate (float): Current water outflow rate (velocity * area of the drain).
    """

    def __init__(self, area_bathtub, area_drain, water_height):
        """
        Initialize the BathtubModel with given parameters.

        Args:
            area_bathtub (float): The cross-sectional area of the bathtub.
            area_drain (float): The cross-sectional area of the drain.
            water_height (float): The initial water height in the bathtub.
        """
        self.area_bathtub = area_bathtub
        self.area_drain = area_drain
        self.water_height = water_height
        self.intial_water_height = water_height
        self.velocity = self.get_velocity()
        self.flow_rate = self.get_flow_rate()
        
    def get_velocity(self):
        """
        Compute the water outflow velocity using Torricelli's law:
        v = sqrt(2 * g * h), where h is the water height.

        Returns:
            float: Computed velocity.
        """
        return jnp.sqrt(2 * 9.81 * self.water_height)
        
    def get_flow_rate(self):
        """
        Compute the outflow (volume/time) from the bathtub
        
        Returns:
            float: Computed flow rate.
        """
        return self.area_drain * self.velocity
    
    def calculate_output(self, u, disturbance):
        """
        Update the water height given an inflow (u) and a disturbance.

        Args:
            u (float): External inflow into the bathtub.
            disturbance (float): Random or unexpected water inflow/outflow factor.

        Returns:
            float: The updated water height.
        """
        dBt = u + disturbance - self.flow_rate

        dHt = dBt / self.area_bathtub

        self.water_height = jnp.maximum(0.001, self.water_height + dHt)

        self.velocity = self.get_velocity()
        self.flow_rate = self.get_flow_rate()
        return self.water_height
        
    def deep_copy(self):
        """
        Create a new BathtubModel instance with the same initial values,
        suitable for running independent simulations.

        Returns:
            BathtubModel: A new instance with identical parameters and initial state.
        """
        return BathtubModel(self.area_bathtub, self.area_drain, self.intial_water_height)
    
    def reset(self):
        """
        Reset the bathtub to its initial water height and recompute velocity/flow rate.
        """
        self.water_height = self.intial_water_height
        self.velocity = self.get_velocity()
        self.flow_rate = self.get_flow_rate()


class CournotModel():
    """
    A simple discrete-time Cournot duopoly model with two firms choosing quantities q1 and q2.
    
    Attributes:
        p_max (float): Intercept of the linear demand curve (max price).
        marginal_cost (float): Common marginal cost for both firms.
        q1 (float): Firm 1's quantity decision.
        q2 (float): Firm 2's quantity decision.
        q (float): Total quantity (q1 + q2).
        market_price (float): Price determined by the linear demand p = p_max - q.
        p1_profit (float): Profit for firm 1, computed as q1 * (market_price - marginal_cost).
        key (PRNGKey): JAX random number generator state.
    """

    def __init__(self, p_max, marginal_cost):
        """
        Initialize the CournotModel with given demand and cost parameters.

        Args:
            p_max (float): Maximum price or intercept of the linear demand.
            marginal_cost (float): Constant marginal cost for both firms.
        """
        self.key = random.PRNGKey(0)
        self.p_max = p_max
        self.marginal_cost = marginal_cost
        
        self.q1, self.q2 = self.get_random_quantity()
        self.q = self.q1 + self.q2
        self.market_price = p_max - self.q
        
    def get_random_quantity(self):
        """
        Generate a random pair of quantities (q1, q2), each in [0,1).

        Returns:
            (float, float): A tuple of random quantities (q1, q2).
        """
        self.key, subkey = random.split(self.key)
        return random.uniform(subkey, shape=(2,))
    
    def deep_copy(self):
        """
        Create a new CournotModel instance with the same initial parameters,
        and reinitialize random quantities.

        Returns:
            CournotModel: A new instance with the same p_max, marginal_cost, but new random q1, q2.
        """
        return CournotModel(self.p_max, self.marginal_cost)
    
    def constraints(self, q):
        """
        Clip the quantity to ensure it remains in the feasible range [0, 1].

        Args:
            q (float): Proposed quantity level.

        Returns:
            float: Clipped quantity within [0, 1].
        """
        return jnp.clip(q, 0, 1)
        
    def calculate_output(self, u, disturbance):
        """
        Update the quantities chosen by each firm and compute Firm 1's profit.

        q1 is adjusted by 'u', q2 by 'disturbance', then both are clipped to [0,1].
        The new total quantity q = q1 + q2 sets the market price as p_max - q.
        Firm 1's profit is q1 * (price - marginal_cost).

        Args:
            u (float): Firm 1's decision shift.
            disturbance (float): Random or external disturbance for Firm 2's decision.

        Returns:
            float: Firm 1's profit (p1_profit).
        """
        self.q1 = self.constraints(self.q1 + u)
        self.q2 = self.constraints(self.q2 + disturbance)
        self.q = self.q1 + self.q2
        self.market_price = self.p_max - self.q
        self.p1_profit = self.q1 * (self.market_price - self.marginal_cost)
        return self.p1_profit
    
    def reset(self):
        """
        Reset the model by re-drawing random quantities for q1 and q2,
        and recomputing total quantity q.
        """
        self.q1, self.q2 = self.get_random_quantity()
        self.q = self.q1 + self.q2  


class FuelTankModel():
    """
    A discrete-time model of a fuel tank with consumption and random disturbances.
    
    Attributes:
        max_capacity (float): Maximum capacity of the fuel tank.
        fuel_consumption_rate (float): Proportional consumption factor of the remaining fuel.
        initial_fuel (float): Initial fuel level (for resets).
        fuel (float): Current fuel level at each timestep.
        key (PRNGKey): JAX random number generator state.
    """

    def __init__(self, fuel_consumption_rate, max_cap, initial_fuel):
        """
        Initialize the FuelTankModel with consumption rate, max capacity, and starting fuel.

        Args:
            fuel_consumption_rate (float): Fractional or proportional consumption rate of the fuel.
            max_cap (float): The maximum capacity of the tank.
            initial_fuel (float): Initial amount of fuel in the tank.
        """
        self.max_capacity = max_cap
        self.fuel_consumption_rate = fuel_consumption_rate
        self.initial_fuel = initial_fuel
        self.fuel = initial_fuel
        self.key = random.PRNGKey(0)
        
    def get_disturbance(self):
        """
        Generate a random disturbance in the range [-0.5, 0.5].
        This could represent, for example, additional consumption due to driving style or external factors.

        Returns:
            float: A random disturbance value in [-0.5, 0.5].
        """
        self.key, subkey = random.split(self.key)
        return random.uniform(subkey, minval=-0.5, maxval=0.5)
    
    def calculate_output(self, u, disturbance):
        """
        Update the fuel tank level given external refueling (u) and a random disturbance.
        
        The consumption is computed as 'fuel_consumption_rate * current_fuel'.
        The net change is u + disturbance - consumption.
        The new fuel level is then clipped between 0 and max_capacity.

        Args:
            u (float): Amount of fuel added at this timestep.
            disturbance (float): Random additional consumption or offset to the net flow.

        Returns:
            float: The updated fuel level.
        """
        consumption = self.fuel_consumption_rate * self.fuel

        dFdt = u + disturbance - consumption
        
        self.fuel = jnp.clip(self.fuel + dFdt, 0, self.max_capacity)
        return self.fuel
    
    def deep_copy(self):
        """
        Create a new FuelTankModel instance with the same parameters,
        but reset to the original initial_fuel level.

        Returns:
            FuelTankModel: A new instance with identical configuration.
        """
        return FuelTankModel(self.fuel_consumption_rate, self.max_capacity, self.initial_fuel)
    
    def reset(self):
        """
        Reset the tank to its initial fuel level.
        """
        self.fuel = self.initial_fuel
