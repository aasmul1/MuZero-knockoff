import jax
from jax import random
import matplotlib.pyplot as plt
import tqdm as tqdm

'''Manage the overall system.
Initialize the plant and controller.
Run simulations for epochs and timesteps.
Handle configurations and logging.'''

from controller import ClassicPIDController, NeuralNetController
from plant import BathtubModel, CournotModel