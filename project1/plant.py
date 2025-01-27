'''Define the plant class and its behavior.
Include subclasses for:
Bathtub model.
Cournot competition.
Your custom plant.
Handle noise generation and state updates'''

import jax.numpy as jnp
from jax import random



# class Plant:
#     def __init__(self):
#         pass

#     def calculate_output(self, u):
#         """Calculate the output of the plant given the input u."""
#         raise NotImplementedError("Subclass must implement abstract method")

#     def reset(self):
#         """Reset the state of the plant."""
#         raise NotImplementedError("Subclass must implement abstract method")

class BathtubModel():
    def __init__(self, area_bathtub, area_drain, water_height):
        self.area_bathtub = area_bathtub
        self.area_drain = area_drain
        self.water_height = water_height
        self.intial_water_height = water_height
        self.velocity = self.get_velocity()
        self.flow_rate = self.get_flow_rate()
        
    def get_velocity(self):
        return jnp.sqrt(2 * 9.81 * self.water_height)
        
    def get_flow_rate(self):
        return self.area_bathtub * self.velocity
    
    def calculate_output(self, u, disturbance):
        dBt = u + disturbance - self.flow_rate
        dHt = dBt / self.area_bathtub
        self.water_height = jnp.maximum(0.001, self.water_height + dHt)
        self.velocity = self.get_velocity()
        self.flow_rate = self.get_flow_rate()
        return self.water_height
        
    def deep_copy(self):
        return BathtubModel(self.area_bathtub, self.area_drain, self.intial_water_height)
    
    def reset(self):
        self.velocity = self.get_velocity()
        self.flow_rate = self.get_flow_rate()
        self.water_height = self.intial_water_height

# class CournotModel(Plant):
#     def __init__(self, p_max, marginal_cost):
#         super().__init__()
#         self.p_max = p_max
#         self.marginal_cost = marginal_cost
#         self.q1, self.q2 = self.get_amount()
#         self.q = self.q1 + self.q2
#         self.actual_price = p_max - self.q
        
#     def get_amount():
#         key = random.PRNGKey(0)
#         random_numbers = random.uniform(key, shape=(2,))
#         return random_numbers[0], random_numbers[1]
    
#     def calculate_output(self, u, disturbance):
#         self.q1 = self.q1 + u
#         self.q2 = self.q2 + disturbance
#         self.q = self.q1 + self.q2
#         self.actual_price = self.p_max - self.q
#         self.p1_profit = self.q1 * (self.actual_price - self.marginal_cost)
#         return self.p1_profit
    
#     def reset(self):
#         self.q1, self.q2 = self.get_amount()
#         self.actual_price = self.p_max - self.q
