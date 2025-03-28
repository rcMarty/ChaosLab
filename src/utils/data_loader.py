import numpy as np


def generate_perceptron_data(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates dataset for perceptron classification based on the line y = 3x + 2.

    :param n: Number of points
    :return: Tuple (X, y) where X are inputs and y are labels. 1 if above, -1 if below
    """
    X = np.random.uniform(-5, 5, (n, 2))
    y_labels = np.where(X[:, 1] > (3 * X[:, 0] + 2), 1, -1)

    return X, y_labels


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
    :return: np.ndarray - Array of arrays of binary patterns (-1, 1).
    """
    return np.random.choice([-1, 1], size=(num_patterns, size * size))


def generate_maze(grid_size: tuple[int, int] = (10, 10), num_obstacles: int = 3) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """
    Generates a simple maze environment with obstacles.

    :param grid_size: Size of the grid (grid_size x grid_size).
    :param num_obstacles: Number of obstacles in the maze.
    :return: Tuple (grid, start, goal) where grid is the maze, start is the starting position, and goal is the goal position.
    """
    grid = np.zeros(grid_size)
    start = (0, 0)
    goal = (grid_size[0] - 1, grid_size[1] - 1)

    # Add obstacles
    for _ in range(num_obstacles):
        x, y = np.random.randint(grid_size, size=2)
        grid[x, y] = 1

    return grid, start, goal


def generate_ifs_data(num_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates data for Iterated Function Systems (IFS).

    :param num_points: Number of points to generate.
    :return: Tuple (X, y) where X are inputs and y are labels.
    """
    # Define the IFS functions
    functions = [
        lambda x, y: (0.5 * x, 0.5 * y),
        lambda x, y: (0.5 * x + 0.5, 0.5 * y),
        lambda x, y: (0.5 * x + 1, 0.5 * y + 1),
        lambda x, y: (0.5 * x + 1.5, 0.5 * y + 1)
    ]

    # Generate points
    X = np.zeros((num_points, 2))
    for i in range(num_points):
        f = np.random.choice(functions)
        X[i] = f(X[i - 1] if i > 0 else (0, 0))

    return X
