import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from src.utils.neural_helpers import Runnable


# 1. Příprava prostředí
class BalanceNetwork:
    """Neuronová síť pro řízení vozíku."""

    def __init__(self, input_size, output_size):
        self.model = self._build_model(input_size, output_size)

    def _build_model(self, input_size, output_size):
        """Vytvoří architekturu neuronové sítě."""
        model = Sequential([
            Dense(24, activation='relu', input_shape=(input_size,)),
            Dense(24, activation='relu'),
            Dense(output_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X, y, epochs=20):
        """Provede trénink sítě na vygenerovaných datech."""
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)


class PoleBalancing(Runnable):
    """Hlavní třída řešící problém balancování tyče."""

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.agent = QLearningAgent(self.env)
        self.nn = BalanceNetwork(input_size=4, output_size=self.env.action_space.n)

    def _generate_training_data(self, samples=10000):
        """Vygeneruje trénovací data z Q-tabulky."""
        X, y = [], []
        for _ in range(samples):
            state = self.agent.discretize_state(self.env.observation_space.sample())
            action = np.argmax(self.agent.q_table[state])
            X.append(state)
            y.append(action)
        return np.array(X), np.array(y)

    def _visualize_training(self, rewards):
        """Vykreslí průběh učení."""
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title("Vývoj odměn během tréninku Q-learning")
        plt.xlabel("Epizoda")
        plt.ylabel("Celková odměna")
        plt.show()

    def _visualize_solution(self):
        """Animuje finální řešení pomocí neuronové sítě."""
        for _ in range(3):  # 3 demonstrační epizody
            obs, _ = self.env.reset()  # Získání pouze pozorování
            state = obs
            total_reward = 0

            while True:
                self.env.render()
                state_disc = self.agent.discretize_state(state)
                action = np.argmax(self.nn.model.predict(np.array([state_disc]), verbose=0))
                obs, reward, done, _, _ = self.env.step(action)
                state = obs
                total_reward += reward

                if done:
                    print(f"Epizoda ukončena s celkovým skóre: {total_reward}")
                    break

        self.env.close()

    @staticmethod
    def run():
        """Hlavní metoda pro spuštění celého pipeline."""
        task = PoleBalancing()

        # Fáze 1: Trénink Q-learning agenta
        rewards = task.agent.train(episodes=3000)
        task._visualize_training(rewards)

        # Fáze 2: Trénink neuronové sítě
        X, y = task._generate_training_data()
        task.nn.train(X, y)

        # Fáze 3: Vizualizace výsledků
        task._visualize_solution()
