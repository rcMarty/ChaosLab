import numpy as np
import matplotlib.pyplot as plt
from src.utils.neural_helpers import Runnable
from src.utils.visualization import save_plot


class FractalTerrain(Runnable):
    """
    Class for generating a 2D fractal terrain using recursive subdivision.
    """

    def __init__(self, recursion_depth: int = 8, initial_perturbation: float = 0.5, decay: float = 0.5):
        self.recursion_depth = recursion_depth
        self.initial_perturbation = initial_perturbation
        self.decay = decay

    def subdivide(self, start: np.ndarray, end: np.ndarray, perturbation: float, depth: int) -> list[np.ndarray]:
        """
        Recursively subdivide a line segment.
        :param start: Starting point (x, y)
        :param end: Ending point (x, y)
        :param perturbation: Current perturbation size
        :param depth: Current depth
        :return: List of points
        """
        if depth == 0:
            return [start, end]

        midpoint = (start + end) / 2
        direction = np.random.choice([-1, 1])
        offset = np.array([0, direction * perturbation])
        midpoint += offset

        left = self.subdivide(start, midpoint, perturbation * self.decay, depth - 1)
        right = self.subdivide(midpoint, end, perturbation * self.decay, depth - 1)

        return left[:-1] + right

    def generate(self) -> np.ndarray:
        """
        Generate terrain points.
        :return: Array of points (n, 2)
        """
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 0.0])

        points = self.subdivide(start, end, self.initial_perturbation, self.recursion_depth)
        return np.array(points)

    def classify_terrain(self, y: np.ndarray) -> np.ndarray:
        """
        Classify terrain into three levels.
        :param y: y-values
        :return: List of colors
        """
        colors = []
        for value in y:
            if value < -0.1:
                colors.append('blue')  # Water
            elif value < 0.1:
                colors.append('green')  # Plains
            else:
                colors.append('brown')  # Mountains
        return np.array(colors)

    @staticmethod
    def run():
        model = FractalTerrain(9, 0.6, 0.45)
        terrain = model.generate()
        x, y = terrain[:, 0], terrain[:, 1]
        colors = model.classify_terrain(y)

        plt.figure(figsize=(12, 6))
        for i in range(len(x) - 1):
            plt.plot(x[i:i + 2], y[i:i + 2], color=colors[i], linewidth=2)

        plt.title("Fractal Terrain Profile")
        plt.xlabel("X")
        plt.ylabel("Elevation")
        plt.grid(True)
        save_plot(plt.gcf(), "fractal_terrain")
        plt.show()
