import numpy as np
from src.utils.data_loader import generate_perceptron_data
from src.utils.neural_helpers import Plot2D, Runnable
from src.utils.visualization import plot_2d_decision_boundary, animate_learning


class Perceptron(Plot2D, Runnable):
    """Simple Perceptron for binary classification."""

    def __init__(self, lr: float = 0.01, epochs: int = 1000):
        """
        Initialize Perceptron model.
        :param lr: Learning rate
        :param epochs: Number of training epochs
        """
        self.lr = lr
        self.epochs = epochs
        self.weights: np.ndarray = None
        self.bias: float = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[int]:
        """Train perceptron using input data X and labels y."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        errors = []

        for _ in range(self.epochs):
            total_error = 0
            for i in range(n_samples):
                y_pred = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if y_pred >= 0 else 0
                update = self.lr * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update
                total_error += int(update != 0)
            errors.append(total_error)

        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input data X."""
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, 0)

    @staticmethod
    def run():
        X, y = generate_perceptron_data(n=100)

        model = Perceptron(lr=0.01, epochs=1000)
        errors = model.fit(X, y)

        animate_learning(errors, title="Error Reduction Over Time")

        plot_2d_decision_boundary(model, X, y)
        
        print(f"Weights: {model.weights}, Bias: {model.bias}")
