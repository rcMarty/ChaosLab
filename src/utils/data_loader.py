import numpy as np


def generate_perceptron_data(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates dataset for perceptron training.
    :param n: Number of points
    :return: Tuple of X (features) and y (labels)
    """
    X = np.random.uniform(-5, 5, (n, 2))  # Random points in range (-5,5)
    y = (X[:, 1] > (3 * X[:, 0] + 2)).astype(int)  # 1 if above, 0 if below
    return X, y


def generate_xor_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Generates dataset for XOR problem.

    :return: Tuple (X, y) where X are inputs and y are labels.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return X, y


def generate_hopfield_patterns(num_patterns: int = 3, size: int = 10) -> np.ndarray:
    """
    Generates binary patterns for Hopfield Network.

    :param num_patterns: Number of patterns to generate.
    :param size: Size of each pattern (size x size grid).
    :return: np.ndarray - Array of binary patterns (-1, 1).
    """
    return np.random.choice([-1, 1], size=(num_patterns, size * size))
