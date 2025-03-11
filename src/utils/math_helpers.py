import numbers

import numpy as np


def sigmoid(x: numbers) -> numbers:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: numbers) -> numbers:
    """Derivative of sigmoid function"""
    return x * (1 - x)


def relu(x: numbers):
    return np.maximum(0, x)


def step_function(x: numbers):
    return np.where(x >= 0, 1, -1)


def normalize(data: np.ndarray, range_min: numbers = 0, range_max: numbers = 1):
    """Normalizuje data do daného rozsahu."""
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * (range_max - range_min) + range_min
