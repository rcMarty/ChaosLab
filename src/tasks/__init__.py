from src.tasks.perceptron import Perceptron
from src.tasks.simple_neural_network import MLP
from src.tasks.hopfield_network import HopfieldNetwork
from src.tasks.q_learn import QLearningAgent
from src.tasks.pole_balancing.q_learnPole import QLearningPoleAgent
from src.tasks.l_system import LSystem
from src.tasks.pole_balancing.pole_balancing import PoleBalancing
from src.tasks.iterated_function_systems import IFS
from src.tasks.mandelbrot_set import MandelbrotSet
from src.tasks.fractal_landscape import FractalTerrain
from src.tasks.logistics_map import LogisticChaos
from src.tasks.forest_fire import ForestFireCA

__all__ = [
    "Perceptron",
    "MLP",
    "HopfieldNetwork",
    "QLearningAgent",
    "LSystem",
    # "QLearningPoleAgent",
    # "PoleBalancing",
    "IFS",
    "MandelbrotSet",
    "FractalTerrain",
    "LogisticChaos",
    "ForestFireCA",
]
