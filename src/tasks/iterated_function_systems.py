from src.utils.neural_helpers import Runnable


class IFS(Runnable):
    """
    Iterated Function Systems (IFS) are a method of constructing fractals

    The resulting fractals are often self-similar and can be generated using a set of affine transformations.
    """

    def __init__(self):
        """
        Initialize IFS parameters.
        """
        self.transformations = []

    @staticmethod
    def run():
        """
        Run the IFS generator.
        """
        from src.utils.data_loader import generate_ifs_data
        from src.utils.visualization import plot_fractal_paths

        # Generate data for IFS
        data = generate_ifs_data(num_points=10000)

        # Plot the generated fractal paths
        plot_fractal_paths(data)
