import gym
import random
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from src.utils.neural_helpers import Runnable


class QLearningPoleAgent(Runnable):
    """Q-learning agent pro problém balancování tyče."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1, episodes: int = 500):
        self.env = gym.make('CartPole-v1')
        self.alpha = alpha  # Rychlost učení
        self.gamma = gamma  # Diskontní faktor
        self.epsilon = epsilon  # Pravděpodobnost explorace
        self.episodes = episodes
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

        # Parametry pro diskretizaci
        self.bins = [
            np.linspace(-2.4, 2.4, 10),  # Pozice vozíku
            np.linspace(-3.0, 3.0, 10),  # Rychlost vozíku
            np.linspace(-0.5, 0.5, 10),  # Úhel tyče
            np.linspace(-2.0, 2.0, 10)  # Úhlová rychlost
        ]

    def discretize_state(self, state):
        """Diskrétizuje spojitý stav do předem definovaných intervalů."""
        discretized = []
        for i in range(len(state)):
            discretized.append(np.digitize(state[i], self.bins[i]))
        return tuple(discretized)

    def choose_action(self, state):
        """Epsilon-greedy výběr akce."""
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def train(self) -> list[float]:
        """Provede trénink agenta a vrátí průběh odměn."""
        rewards = []
        for episode in range(self.episodes):
            state = self.discretize_state(self.env.reset()[0])
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                obs, reward, done, _, _ = self.env.step(action)
                next_state = self.discretize_state(obs)

                # Aktualizace Q-tabulky
                old_value = self.q_table[state][action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.q_table[state][action] = new_value

                state = next_state
                total_reward += reward

            rewards.append(total_reward)

            if episode % 100 == 0:
                print(f"Epizoda {episode}: Odměna = {total_reward}")

        return rewards

    @staticmethod
    def visualize(rewards):
        """Vykreslí průběh učení."""
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title("Vývoj odměn během tréninku")
        plt.xlabel("Epizoda")
        plt.ylabel("Celková odměna")
        plt.show()

    @staticmethod
    def run():
        """Hlavní metoda pro spuštění celého procesu."""
        agent = QLearningPoleAgent(episodes=1000)
        rewards = agent.train()
        agent.visualize(rewards)

        # Ukázka finálního výkonu
        state = agent.env.reset()[0]
        done = False
        while not done:
            agent.env.render()
            action = np.argmax(agent.q_table[agent.discretize_state(state)])
            state, _, done, _, _ = agent.env.step(action)

        agent.env.close()
