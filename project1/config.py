'''Store all configuration parameters.
Include options like plant type, controller type, learning rate, number of timesteps, etc.'''
consys_params = {
    "epochs": 70, 
    "timesteps": 25, 
    "disturbance_range": {-0.01, 0.01},
    "controller": "NeuralNetController", 
    "plant": "FuelTank",
    "learning_rate": 0.01
    }

neural_network_params = {
    "hidden_layers": [16, 4, 6, 6],
    "activation_layers": ["relu", "relu", "tanh", "sigmoid"],
    "weight_range": {-1.0, 1.0},
    "bias_range": {0.0, 1.0}   
    }

bathtub_params = {
    "area_bathtub": 3.0, 
    "area_drain": 0.01, 
    "water_height": 15.0,
    "target": 10.0
    }

cournot_params = {
    "p_max": 5.0, 
    "marginal_cost": 0.1,
    "target": 0.5
}

fuelTank_params = {
    "consumption_rate": 0.1, 
    "initial_fuel": 5.0, 
    "max_cap": 10.0,
    "target": 9.0
}


