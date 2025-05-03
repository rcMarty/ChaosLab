import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from src.utils.neural_helpers import Runnable
from src.utils.visualization import save_plot
from tqdm import tqdm


class LogisticChaos(Runnable):
    """
    Logistic map simulation to illustrate deterministic chaos with neural network prediction.
    """

    def __init__(self, a_min=0.0, a_max=4.0, a_steps=1000, n_iter=100, burn_in=100):
        """
        Initialize the LogisticChaos class with parameters for the logistic map.
        :param a_min: Minimum value of the control parameter 'a'.
        :param a_max: Maximum value of the control parameter 'a'.
        :param a_steps: Number of steps to divide the range of 'a'.
        :param n_iter: Number of iterations to generate the sequence.
        :param burn_in: Number of iterations to discard before collecting data.
        """

        self.a_values = np.linspace(a_min, a_max, a_steps)
        self.n_iter = n_iter
        self.burn_in = burn_in

    @staticmethod
    def logistic_map(x, a):
        return a * x * (1 - x)

    def generate_sequence(self, x0, r) -> np.ndarray:
        """
        Generate logistic map sequence.
        :return: numpy array with sequence values
        """
        x = np.zeros(self.n_iter)
        x[0] = x0
        for i in range(1, self.n_iter):
            x[i] = r * x[i - 1] * (1 - x[i - 1])
        return x

    def generate_bifurcation_data(self) -> np.ndarray:
        """
        Generate bifurcation data for the logistic map.
        :return: numpy array with bifurcation data
        """
        xs = []
        for a in self.a_values:
            x = 0.5
            for _ in range(self.burn_in):  # Burn-in period
                x = self.logistic_map(x, a)
            for _ in range(self.n_iter):  # Collect data
                x = self.logistic_map(x, a)
                xs.append((a, x))
        return np.array(xs)

    @staticmethod
    def plot_bifurcation(data, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(data[:, 0], data[:, 1], ',k', alpha=0.25)
        plt.title(title)
        plt.xlabel("a")
        plt.ylabel("x")
        save_plot(plt.gcf(), filename)
        plt.show()

    @staticmethod
    def plot_bifurcation_two(actual_data, predicted_data, title, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_data[:, 0], actual_data[:, 1], ',k', alpha=0.25, label="Actual")
        plt.plot(predicted_data[:, 0], predicted_data[:, 1], ',r', alpha=0.5, label="Predicted")
        plt.title(title)
        plt.xlabel("a")
        plt.ylabel("x")
        plt.legend()
        save_plot(plt.gcf(), filename)
        plt.show()

    @staticmethod
    def train_neural_network(data):
        """
        Train a neural network to predict the bifurcation diagram.
        :param data: numpy array with bifurcation data
        :return: trained model
        """
        x_train = data[:, 0].reshape(-1, 1)  # a values
        y_train = data[:, 1]  # x values

        model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
        return model

    def predict_bifurcation(self, model):
        """
        Predict the bifurcation diagram using the trained neural network.
        :param model: trained neural network model
        :return: numpy array with predicted bifurcation data
        """
        predictions = []
        for a in tqdm(self.a_values, desc="Predicting bifurcation", unit="a-value"):
            x = 0.5
            for _ in range(self.burn_in):  # Burn-in period
                x = self.logistic_map(x, a)
            for _ in range(self.n_iter):  # Predict data
                x = model.predict(np.array([[a]]), verbose=0)[0][0]
                predictions.append((a, x))
        return np.array(predictions)

    @staticmethod
    def run():
        chaos = LogisticChaos()
        sequence = chaos.generate_sequence(0.5, 3.9)

        # Plot time series
        plt.figure(figsize=(10, 5))
        plt.plot(sequence)
        plt.title("Logistic Map Time Series")
        plt.xlabel("Iteration")
        plt.ylabel("x")
        plt.grid(True)
        plt.legend()
        save_plot(plt.gcf(), "logistic_sequence")
        plt.show()

        data = chaos.generate_bifurcation_data()

        chaos.plot_bifurcation(data, "Actual Bifurcation Diagram", "logistic_actual_bifurcation")

        model = chaos.train_neural_network(data)
        print("Neural network trained.")
        predicted_data = chaos.predict_bifurcation(model)
        print("Predicted Bifurcation Diagram.")
        chaos.plot_bifurcation_two(data, predicted_data, "Predicted Bifurcation Diagram", "logistic_predicted_bifurcation")
