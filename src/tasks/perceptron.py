import numpy as np
from src.utils.data_loader import generate_perceptron_data
from src.utils.neural_helpers import Plot2DBoundary, Runnable
from src.utils.visualization import plot_2d_decision_boundary, animate_learning, visualize_data, save_plot


class Perceptron(Plot2DBoundary, Runnable):
    """Simple Perceptron for binary classification."""

    def __init__(self, lr: float = 0.01, epochs: int = 1000):
        """
        Initialize Perceptron model.
        :param lr: Learning rate
        :param epochs: Number of training epochs
        """
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """
        Train perceptron using input data X and labels y.

        :param X: Input data, shape (n_samples, n_features)
        :param y: Labels, shape (n_samples,)
        :return: List of errors during training in each epoch
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        errors = []

        for epoch in range(self.epochs):
            total_error = 0
            correct_predictions = 0

            for i in range(n_samples):
                y_pred = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if y_pred >= 0 else -1  # Binary classification
                update = self.lr * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update
                total_error += abs(update)

                if y_pred == y[i]:
                    correct_predictions += 1

            errors.append(total_error / n_samples)

            # Print accuracy every 100 epochs
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                accuracy = correct_predictions / n_samples
                print(f"Epoch {epoch}: Accuracy = {accuracy * 100:.2f}%")

        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input data X."""
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, -1)

    def get_edge_for_plot(self):
        """
        Vrátí rovnici přímky pro vizualizaci.
        :return: lambda funkce pro výpočet y
        """
        if self.weights is None or self.weights[1] == 0:
            return lambda x: 0  # Avoid division by zero

        def line(x):
            return -(self.bias + self.weights[0] * x) / self.weights[1]

        return line

    @staticmethod
    def run():
        X, y = generate_perceptron_data(n=100)

        # Normalize input data
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        model = Perceptron(lr=0.01, epochs=1000)
        errors = model.fit(X, y)

        animate_learning(errors, title="Perceptron Training Progress")

        # Vizuální vykreslení bodů
        fig, ax = visualize_data(X, y, show=False)

        # Originální přímka y = 3x + 2
        x_vals = np.linspace(-2, 2, 100)
        y_vals_original = 3 * x_vals + 2
        ax.plot(x_vals, y_vals_original, label="Original Line: y = 3x + 2", color="red", linestyle="--")

        # Naučená hranice perceptronu
        learned_line = model.get_edge_for_plot()
        y_vals_learned = learned_line(x_vals)
        ax.plot(x_vals, y_vals_learned, label="Learned Decision Boundary", color="blue")

        ax.legend()
        fig.show()
        save_plot(fig, "Perceptron")

        # Final accuracy calculation
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y)
        print(f"Final Accuracy: {accuracy * 100:.2f}%")

        print(f"Weights: {model.weights}, Bias: {model.bias}")
