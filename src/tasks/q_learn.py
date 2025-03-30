import numpy as np
import random
import matplotlib.pyplot as plt
from src.utils.neural_helpers import Runnable
from src.utils.data_loader import generate_maze
from src.utils.visualization import visualize_grid, animate_learning, save_plot, visualize_maze


class QLearningAgent(Runnable):
    """Q-learning agent solving a grid-based maze to find the cheese."""

    def __init__(self, grid_size: tuple[int, int] = (5, 5), alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1, episodes: int = 500, obstacles: int = 5):
        """
        Initialize Q-learning agent parameters.
        :param grid_size: a tuple(int, int) - size of the grid
        :param alpha: learning rate
        :param gamma: discount factor
        :param epsilon: exploration rate
        :param episodes: number of training episodes
        :param obstacles: number of obstacles in the maze
        """
        self.grid_size = grid_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.episodes = episodes
        self.actions = ["up", "down", "left", "right"]
        self.q_table = np.zeros((*grid_size, len(self.actions)))  # Initialize Q-table
        self.maze, self.start, self.goal = generate_maze(grid_size, num_obstacles=obstacles)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.actions)))  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def train(self) -> list[float]:
        """
        Train the Q-learning agent and return rewards per episode.

        :return: List of total rewards collected per episode.
        """
        rewards_per_episode = []

        for episode in range(self.episodes):
            state = self.start
            total_reward = 0

            while state != self.goal:
                action = self.choose_action(state)
                next_state, reward = self.step(state, action)

                # Update Q-table
                self.q_table[state][action] += self.alpha * (
                        reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
                )

                state = next_state
                total_reward += reward

            rewards_per_episode.append(total_reward)  # Track cumulative reward per episode

            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}")

        return rewards_per_episode

    def step(self, state, action):
        """Execute action and return new state and reward."""
        x, y = state
        if self.actions[action] == "up":
            x = max(0, x - 1)
        elif self.actions[action] == "down":
            x = min(self.grid_size[0] - 1, x + 1)
        elif self.actions[action] == "left":
            y = max(0, y - 1)
        elif self.actions[action] == "right":
            y = min(self.grid_size[1] - 1, y + 1)

        new_state = (x, y)
        reward = -1 * (abs(x - self.goal[0]) + abs(y - self.goal[1]))  # Penalizing longer paths
        # Define rewards
        if new_state == self.goal:
            reward = 10000
        elif self.maze[x, y] == 1:
            reward = -10000

        return new_state, reward

    def show_path(self):
        """
        Show the best path found by the Q-learning agent.
        This function visualizes the best path on the maze grid.
        :return:
        """
        # print best path
        path = []
        state = self.start
        while state != self.goal:
            action = np.argmax(self.q_table[state])
            path.append(state)
            state, _ = self.step(state, action)
        path.append(self.goal)
        print("Best path found by Q-learning agent:", path)
        # Visualize the best path
        fig, ax = plt.subplots(figsize=(11, 11))
        ax.imshow(self.maze, cmap="gray", alpha=0.5)
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], marker="o", color="blue", label="Best Path")
        ax.scatter(self.start[1], self.start[0], marker="o", color="green", label="Start")
        ax.scatter(self.goal[1], self.goal[0], marker="X", color="red", label="Cheese 🧀")
        ax.legend()
        plt.title("Best Path Found by Q-Learning Agent")
        save_plot(fig, "q_best_path.png")
        plt.show()

    @staticmethod
    def run():

        agent = QLearningAgent(grid_size=(10, 10), episodes=500, obstacles=10)

        visualize_grid(agent.maze, title="q_Learning Maze", save=True)
        visualize_maze(agent.maze, agent.start, agent.goal)

        # Train the Q-learning agent and track learning progress
        rewards_per_episode = agent.train()
        # print("revards per episode:", rewards_per_episode)
        animate_learning(rewards_per_episode, title="Q-Learning Progress")

        # Save Q-table as a heatmap
        fig, ax = plt.subplots(figsize=(11, 11))
        ax.imshow(np.max(agent.q_table, axis=2), cmap="coolwarm")
        plt.title("Final Q-Table Heatmap")
        save_plot(fig, "q_table.png")
        plt.show()

        # Show the best path found by the Q-learning agent
        agent.show_path()
