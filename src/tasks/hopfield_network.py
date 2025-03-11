from abc import ABC

import numpy as np

from src.utils.math_helpers import step_function
from src.utils.neural_helpers import Runnable, Plot2D


class HopfieldNetwork(Plot2D, Runnable, ABC):
    """
    Hopfield Network for pattern storage and recall.
    """

    def __init__(self, size: int):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns: np.ndarray) -> None:
        """
        Trains the network using Hebbian learning.

        :param patterns: np.ndarray - Set of binary patterns (-1, 1) to be stored.
        """
        assert np.all(np.isin(patterns, [-1, 1])), "Patterns must contain only -1 and 1"
        self.weights.fill(0)

        for p in patterns:
            self.weights += np.outer(p, p)

        np.fill_diagonal(self.weights, 0)  # No self-connections

    def recall(self, pattern: np.ndarray, steps: int = 10) -> np.ndarray:
        """
        Recalls a pattern using iterative updates.

        :param pattern: np.ndarray - Initial noisy pattern.
        :param steps: int - Number of iterations for recall.
        :return: np.ndarray - Recalled pattern.
        """

        indices = np.arange(len(pattern))
        for _ in range(steps):
            np.random.shuffle(indices)
            for i in indices:
                pattern[i] = step_function(np.dot(self.weights[i], pattern))
        return pattern

    @staticmethod
    def run():
        from src.utils.data_loader import generate_hopfield_patterns
        from src.utils.visualization import visualize_grid

        patterns = generate_hopfield_patterns(num_patterns=3, size=10)

        model = HopfieldNetwork(size=patterns.shape[1])
        model.train(patterns)

        noisy_pattern = patterns[0].copy()
        noisy_pattern[:3] *= -1  # Poškození prvních 3 neuronů

        restored_pattern = model.recall(noisy_pattern, steps=10)

        visualize_grid(patterns[0].reshape(10, 10), title="Original Pattern")
        visualize_grid(noisy_pattern.reshape(10, 10), title="Noisy Pattern")
        visualize_grid(restored_pattern.reshape(10, 10), title="Restored Pattern")
