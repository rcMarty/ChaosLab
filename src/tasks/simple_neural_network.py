import numpy as np

from src.utils.data_loader import generate_xor_data
from src.utils.neural_helpers import Plot2D, Runnable
from src.utils.visualization import animate_learning, plot_2d_decision_boundary


class MLP(Plot2D, Runnable):
    """
    Simple Multi-Layer Perceptron (MLP) for XOR classification.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 0.1):
        self.lr = lr
        # Inicializace vah a biasů
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """Backward propagation with gradient descent"""
        m = y.shape[0]
        error = self.a2 - y
        dW2 = np.dot(self.a1.T, error * self.sigmoid_derivative(self.a2))
        db2 = np.sum(error * self.sigmoid_derivative(self.a2), axis=0, keepdims=True)

        dW1 = np.dot(X.T, (np.dot(error * self.sigmoid_derivative(self.a2), self.W2.T) * self.sigmoid_derivative(self.a1)))
        db1 = np.sum((np.dot(error * self.sigmoid_derivative(self.a2), self.W2.T) * self.sigmoid_derivative(self.a1)), axis=0, keepdims=True)

        # Update váhy
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5000) -> list:
        """Train the MLP"""
        errors = []
        for i in range(epochs):
            self.forward(X)
            self.backward(X, y)
            loss = np.mean(np.abs(self.a2 - y))
            errors.append(loss)
        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.round(self.forward(X))

    @staticmethod
    def run():
        X, y = generate_xor_data()
        # Trénování modelu
        model = MLP(input_size=2, hidden_size=4, output_size=1, lr=0.1)
        errors = model.train(X, y)

        # Vykreslení chyby
        animate_learning(errors, title="Error Reduction Over Time")

        # Vykreslení rozhodovací hranice
        plot_2d_decision_boundary(model, X, y)

        # Výpis vah
        print(f"W1: {model.W1}, b1: {model.b1}")
        print(f"W2: {model.W2}, b2: {model.b2}")
        print("Predictions:")
        print(model.predict(X))
