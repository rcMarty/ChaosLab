import numpy as np
from numpy import ndarray

from src.utils.data_loader import generate_xor_data
from src.utils.math_helpers import sigmoid, sigmoid_derivative
from src.utils.neural_helpers import Plot2DBoundary, Runnable
from src.utils.visualization import animate_learning, plot_2d_decision_boundary, visualize_data


class MLP(Plot2DBoundary, Runnable):
    """
    Simple Multi-Layer Perceptron (MLP) for XOR classification.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float = 0.1):
        """
        Initialize MLP model.
        :param input_size:  Number of input features
        :param hidden_size:  Number of hidden units
        :param output_size:  Number of output units
        :param lr:  Learning rate
        """
        self.lr = lr
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)  # weight matrix from input to hidden layer
        self.b1 = np.zeros((1, hidden_size))  # bias vector for hidden layer
        self.W2 = np.random.randn(hidden_size, output_size)  # weight matrix from hidden to output layer
        self.b2 = np.zeros((1, output_size))  # bias vector for output layer

        # Forward propagation variables
        self.z1: np.ndarray = ndarray([])  # input to hidden layer
        self.a1: np.ndarray = ndarray([])  # activation of hidden layer
        self.z2: np.ndarray = ndarray([])  # hidden to output layer
        self.a2: np.ndarray = ndarray([])  # activation of output layer

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """Backward propagation with gradient descent"""
        error = self.a2 - y
        dW2 = np.dot(self.a1.T, error * sigmoid_derivative(self.a2))
        db2 = np.sum(error * sigmoid_derivative(self.a2), axis=0, keepdims=True)

        dW1 = np.dot(X.T, (np.dot(error * sigmoid_derivative(self.a2), self.W2.T) * sigmoid_derivative(self.a1)))
        db1 = np.sum((np.dot(error * sigmoid_derivative(self.a2), self.W2.T) * sigmoid_derivative(self.a1)), axis=0, keepdims=True)

        # Update weghts and biases
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

        animate_learning(errors, title="Error Reduction Over Time", animate_time=0.00000001)  # Vykreslení chyby

        plot_2d_decision_boundary(model, X, y, "XOR problem boundary", True)  # Vykreslení rozhodovací hranice

        # Výpis vah
        print(f"W1: {model.W1},\n b1: {model.b1}")
        print(f"W2: {model.W2},\n b2: {model.b2}")
        print("Predictions:")
        print(model.predict(X))
