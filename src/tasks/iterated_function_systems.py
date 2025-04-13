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
                (np.array([[0.00, 0.00, 0.01], [0.00, 0.26, 0.00], [0.00, 0.00, 0.05]]), np.array([0.00, 0.00, 0.00]), 0.25),
                (np.array([[0.20, -0.26, -0.01], [0.23, 0.22, -0.07], [0.07, 0.00, 0.24]]), np.array([0.00, 0.80, 0.00]), 0.25),
                (np.array([[-0.25, 0.28, 0.01], [0.26, 0.24, -0.07], [0.07, 0.00, 0.24]]), np.array([0.00, 0.22, 0.00]), 0.25),
                (np.array([[0.85, 0.04, -0.01], [-0.04, 0.85, 0.09], [0.00, 0.08, 0.84]]), np.array([0.00, 0.80, 0.00]), 0.25),
            ],
            "fern2d": [
                (np.array([[0.0, 0.0, 0.0], [0.0, 0.16, 0.0], [0.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0]), 0.01),
                (np.array([[0.85, 0.04, 0.0], [-0.04, 0.85, 0.0], [0.0, 0.0, 0.0]]), np.array([0.0, 1.6, 0.0]), 0.85),
                (np.array([[0.2, -0.26, 0.0], [0.23, 0.22, 0.0], [0.0, 0.0, 0.0]]), np.array([0.0, 1.6, 0.0]), 0.07),
                (np.array([[-0.15, 0.28, 0.0], [0.26, 0.24, 0.0], [0.0, 0.0, 0.0]]), np.array([0.0, 0.44, 0.0]), 0.07),
            ],
            "custom": [
                (np.array([[0.05, 0.00, 0.00], [0.00, 0.60, 0.00], [0.00, 0.00, 0.05]]), np.array([0.00, 0.00, 0.00]), 0.25),
                (np.array([[0.45, -0.22, 0.22], [0.22, 0.45, 0.22], [-0.22, 0.22, -0.45]]), np.array([0.00, 1.00, 0.00]), 0.25),
                (np.array([[-0.45, 0.22, -0.22], [0.22, 0.45, 0.22], [0.22, -0.22, 0.45]]), np.array([0.00, 1.25, 0.00]), 0.25),
                (np.array([[0.49, -0.08, 0.08], [0.08, 0.49, 0.08], [0.08, -0.08, 0.49]]), np.array([0.00, 2.00, 0.00]), 0.25),
            ]
        }

        # Default to "fern"
        self.transforms = self.transform_map["fern2d"]

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
        :return: Tuple of X, Y, and Z coordinates of generated points
        """
        # Initialize a random starting point in 3D space
        x, y, z = np.random.uniform(-1, 1), np.random.uniform(0, 3), np.random.uniform(-1, 1)
        points = []

        # Extract probabilities and transformations
        probs = [t[2] for t in self.transforms]
        transforms = [t[:2] for t in self.transforms]

        for _ in range(self.iterations):
            # Randomly select a transformation based on probabilities
            A, b = transforms[np.random.choice(len(transforms), p=probs)]

            # Apply the affine transformation
            vec = np.dot(A, np.array([x, y, z])) + b
            x, y, z = vec[0], vec[1], vec[2]
            points.append((x, y, z))

        # Convert points to a NumPy array
        points = np.array(points)
        return np.column_stack((points[:, 0], points[:, 1], points[:, 2]))

    @staticmethod
    def run():
        model = IFS(100000)
        nppoints = model.generate()
        x, y, z = nppoints[:, 0], nppoints[:, 1], nppoints[:, 2]
        animate_fractal_generation(x, y)
        plot_fractal(x, y, "IFS Fractal", save=True)
