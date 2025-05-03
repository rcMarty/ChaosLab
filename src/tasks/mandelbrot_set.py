import numpy as np
import matplotlib.pyplot as plt
from src.utils.neural_helpers import Runnable
from src.utils.visualization import save_plot


class MandelbrotSet(Runnable):
    """
    Generates and visualizes the Mandelbrot set.
    """

    def __init__(self, width: int = 800, height: int = 800, max_iter: int = 100, zoom: float = 1.0, x_center: float = -0.5, y_center: float = 0.0, c: complex = complex(-0.7, 0.27015)):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.zoom = zoom
        self.x_center = x_center
        self.y_center = y_center

        self.c = c

    def generate_fractal(self, fractal_type: str = "mandelbrot") -> np.ndarray:
        """
        Generate a fractal (Mandelbrot or Julia set).
        :param fractal_type: "mandelbrot" or "julia"
        :return: 2D array representing the fractal
        """
        if fractal_type == "mandelbrot":
            x = np.linspace(self.x_center - 2.0 / self.zoom, self.x_center + 2.0 / self.zoom, self.width)
            y = np.linspace(self.y_center - 2.0 / self.zoom, self.y_center + 2.0 / self.zoom, self.height)
            X, Y = np.meshgrid(x, y)
            C = X + 1j * Y
            Z = np.zeros_like(C)
        elif fractal_type == "julia":
            x = np.linspace(-1.5 / self.zoom, 1.5 / self.zoom, self.width)
            y = np.linspace(-1.5 / self.zoom, 1.5 / self.zoom, self.height)
            X, Y = np.meshgrid(x, y)
            Z = X + 1j * Y
            C = self.c
        else:
            raise ValueError("Invalid fractal type. Use 'mandelbrot' or 'julia'.")

        output = np.zeros(Z.shape, dtype=int)

        for i in range(self.max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask] if fractal_type == "mandelbrot" else Z[mask] ** 2 + self.c
            output[mask] = i

        return output

    @staticmethod
    def run():
        model = MandelbrotSet()
        fractal = model.generate_fractal("mandelbrot")

        plt.figure(figsize=(10, 10))
        plt.imshow(fractal, cmap="inferno", extent=(-2, 2, -2, 2))
        plt.colorbar(label="Number of iterations")
        plt.title("Mandelbrot Set")
        save_plot(plt.gcf(), "tea_mandelbrot_set")
        plt.show()

        fractal = model.generate_fractal("julia")
        plt.figure(figsize=(10, 10))
        plt.imshow(fractal, cmap="plasma", extent=(-1.5, 1.5, -1.5, 1.5))
        plt.colorbar(label="Number of iterations")
        plt.title("Julia Set")
        save_plot(plt.gcf(), "tea_julia_set")
        plt.show()
