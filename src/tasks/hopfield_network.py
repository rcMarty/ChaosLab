import numpy as np

from src.utils.math_helpers import step_function
from src.utils.neural_helpers import Runnable


class HopfieldNetwork(Runnable):
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
        np.fill_diagonal(self.weights, 0)  # Zero out self-connections
        self.weights /= patterns.shape[1]  # normalize

    def recall(self, pattern: np.ndarray, steps: int = 10, async_update: bool = True) -> np.ndarray:
        """
        Recalls a pattern using iterative updates.

        :param pattern: np.ndarray - Initial noisy pattern.
        :param steps: int - Number of iterations for recall.
        :param async_update: bool - If True, update neurons asynchronously.
        :return: np.ndarray - Recalled pattern.
        """

        pattern = pattern.copy()
        indices = np.arange(len(pattern))
        for _ in range(steps):
            if async_update:
                np.random.shuffle(indices)
                for i in indices:
                    pattern[i] = step_function(np.dot(self.weights[i], pattern))
            else:
                pattern = step_function(np.dot(self.weights, pattern))
        return pattern

    @staticmethod
    def run():
        from src.utils.data_loader import generate_hopfield_patterns
        from src.utils.visualization import visualize_grid

        size = 10
        patterns = generate_hopfield_patterns(num_patterns=7, size=size)
        print("real orig Pattern: ", patterns)

        model = HopfieldNetwork(size=patterns.shape[1])
        model.train(patterns)

        # Increase noise by flipping random pixels
        noisy_pattern = patterns[0].copy()
        num_flips = int(size * size * 0.3)  # Flip 30% of pixels
        flip_indices = np.random.choice(len(noisy_pattern), num_flips, replace=False)
        noisy_pattern[flip_indices] *= -1

        restored_pattern = model.recall(noisy_pattern, steps=10)

        print("Original Pattern:  ", patterns[0])
        print("Noisy Pattern:     ", noisy_pattern)
        print("Restored Pattern:  ", restored_pattern)

        # Visualize all three states
        visualize_grid(patterns[0].reshape(size, size), title="hop_Original Pattern", save=True)
        visualize_grid(noisy_pattern.reshape(size, size), title="hop_Noisy Pattern", save=True)
        visualize_grid(restored_pattern.reshape(size, size), title="hop_Restored Pattern", save=True)
