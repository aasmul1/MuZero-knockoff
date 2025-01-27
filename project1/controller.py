#traditional PID controller
'''Define the base controller class.
Subclass into:
Traditional PID controller: Update using the three parameters (kp, ki, kd).
AI-based controller: Neural network implementation using JAX'''

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

# class NeuralNetController():
    

#     def control(self, error):
#         return self.model.predict(error)
    

