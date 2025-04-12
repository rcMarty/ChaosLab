import numpy as np
from src.utils.neural_helpers import Runnable
from src.utils.visualization import animate_fractal_generation, plot_fractal_paths, plot_fractal


class IFS(Runnable):
    """
    Iterated Function System (IFS) for generating fractals using affine transformations.
    """

    def __init__(self, iterations: int = 10000):
        self.iterations = iterations
        # Define affine transformations (A, b, and probability)
        self.transform_map = {
            "fern": [
                (np.array([[0.0, 0.0], [0.0, 0.16]]), np.array([0.0, 0.0]), 0.01),
                (np.array([[0.85, 0.04], [-0.04, 0.85]]), np.array([0.0, 1.6]), 0.85),
                (np.array([[0.2, -0.26], [0.23, 0.22]]), np.array([0.0, 1.6]), 0.07),
                (np.array([[-0.15, 0.28], [0.26, 0.24]]), np.array([0.0, 0.44]), 0.07),
            ],
            "triangle": [
                (np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([0.0, 0.0]), 0.33),
                (np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([1.0, 0.0]), 0.33),
                (np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([0.5, 0.866]), 0.34),
            ],
            "square": [
                (np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([0.0, 0.0]), 0.25),
                (np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([0.5, 0.0]), 0.25),
                (np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([0.0, 0.5]), 0.25),
                (np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([0.5, 0.5]), 0.25),
            ],
            "custom": [
                (np.array([[0.05, 0.00, 0.00], [0.00, 0.60, 0.00], [0.00, 0.00, 0.05]]), np.array([0.00, 0.00, 0.00]), 0.25),
                (np.array([[0.45, -0.22, 0.22], [0.22, 0.45, 0.22], [-0.22, 0.22, -0.45]]), np.array([0.00, 1.00, 0.00]), 0.25),
                (np.array([[-0.45, 0.22, -0.22], [0.22, 0.45, 0.22], [0.22, -0.22, 0.45]]), np.array([0.00, 1.25, 0.00]), 0.25),
                (np.array([[0.49, -0.08, 0.08], [0.08, 0.49, 0.08], [0.08, -0.08, 0.49]]), np.array([0.00, 2.00, 0.00]), 0.25),
            ]
        }

        # Default to "fern"
        self.transforms = self.transform_map["fern"]

    def set_transforms(self, name: str) -> None:
        """
        Set the active transformations based on the provided name.
        :param name: Name of the transformation set
        """
        if name in self.transform_map:
            self.transforms = self.transform_map[name]
        else:
            raise ValueError(f"Transformation '{name}' not found.")

    def generate(self) -> np.ndarray:
        """
        Generate IFS fractal points.
        :return: Tuple of X and Y coordinates of generated points
        """
        x, y = np.random.uniform(-1, 1), np.random.uniform(0, 2)
        points = []

        probs = [t[2] for t in self.transforms]
        transforms = [t[:2] for t in self.transforms]

        for _ in range(self.iterations):
            A, b = transforms[np.random.choice(len(transforms), p=probs)]
            vec = np.dot(A, np.array([x, y])) + b
            x, y = vec[0], vec[1]
            points.append((x, y))

        points = np.array(points)
        return np.column_stack((points[:, 0], points[:, 1]))

    @staticmethod
    def run():
        model = IFS()
        nppoints = model.generate()
        x, y = nppoints[:, 0], nppoints[:, 1]
        animate_fractal_generation(x, y)
        plot_fractal(x, y, "IFS Fractal", save=True)
