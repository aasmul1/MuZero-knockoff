#traditional PID controller
'''Define the base controller class.
Subclass into:
Traditional PID controller: Update using the three parameters (kp, ki, kd).
AI-based controller: Neural network implementation using JAX'''

class Controller:
    def __init__(self, kp, ki, kd):
        pass

    def control(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class ClassicPIDController(Controller):
    def __init__(self, kp, ki, kd):
        super().__init__(kp, ki, kd)
#neural-net-based controller

class NeuralNetController(Controller):
    def __init__(self, model):
        self.model = model

    def control(self, error):
        return self.model.predict(error)
    

