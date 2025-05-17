import gymnasium as gym
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

from matplotlib import animation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from src.utils.neural_helpers import Runnable

matplotlib.use('TkAgg')


class DQNAgent(Runnable):
    """
    Deep Q-Learning agent for the CartPole balancing task.
    """

    def __init__(self, state_size, action_size):
        """
        Initialize the DQN agent with the given state and action sizes.
        :param state_size : Size of the state space.
        :param action_size : Size of the action space.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @staticmethod
    def train_cartpole(episodes=300):
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        batch_size = 32
        rewards = []

        for e in range(episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            for time in range(500):
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"Episode {e + 1}/{episodes} - Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                    break
            rewards.append(total_reward)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Plot the reward progress
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('CartPole-v1 using DQN (Gymnasium 1.1.1)')
        plt.savefig('results/cartpole_train.png')
        plt.show()

        return agent

    @staticmethod
    def visualize_agent(agent, episodes=1):
        env = gym.make('CartPole-v1', render_mode="rgb_array")
        state_size = env.observation_space.shape[0]

        for ep in range(episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            done = False
            frames = []

            while not done:
                frame = env.render()  # returns RGB array
                frames.append(frame)

                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = np.reshape(next_state, [1, state_size])

            print(f"Visualization Episode {ep + 1}: Score = {total_reward}")

            # Save GIF
            fig = plt.figure()
            plt.axis('off')
            ims = [[plt.imshow(f, animated=True)] for f in frames]
            ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
            ani.save(f"results/cartpole_episode_{ep + 1}.gif", writer="pillow", fps=20)
            plt.close(fig)

        env.close()

    @staticmethod
    def run(episodes=500):
        trained_agent = DQNAgent.train_cartpole()
        DQNAgent.visualize_agent(trained_agent, episodes=30)
