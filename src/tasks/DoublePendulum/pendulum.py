import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

from src.utils.neural_helpers import Runnable

matplotlib.use('TkAgg')


class DoublePendulum(Runnable):
    def __init__(self, ax, l1=1, l2=1, m1=1, m2=1, theta1=np.pi / 2, theta2=np.pi / 2,
                 color='blue', theta1_dot=0, theta2_dot=0, g=9.81):
        self.ax = ax
        self.l1, self.l2 = l1, l2
        self.m1, self.m2 = m1, m2
        self.theta1, self.theta2 = theta1, theta2
        self.theta1_dot, self.theta2_dot = theta1_dot, theta2_dot
        self.g = g
        self.color = color
        self.max_angular_velocity = 20

        # Trail parameters
        self.trail = []
        self.trail_length = 50
        self.trail_size = m2 * 25

        # Initialize positions
        self.update_positions()

        # Create pendulum elements
        self.rod1, = ax.plot([0, self.x1], [0, self.y1], color=color, lw=2)
        self.rod2, = ax.plot([self.x1, self.x2], [self.y1, self.y2], color=color, lw=2)
        self.mass1 = ax.scatter([self.x1], [self.y1], s=m1 * 50, color=color, zorder=3)
        self.mass2 = ax.scatter([self.x2], [self.y2], s=m2 * 50, color=color, zorder=3)

        # Trail visualization
        self.trail_scatter = ax.scatter(
            [], [],
            s=self.trail_size,
            color=self.color,
            alpha=0.3,
            zorder=2
        )

    def update_positions(self):
        self.x1 = self.l1 * np.sin(self.theta1)
        self.y1 = -self.l1 * np.cos(self.theta1)
        self.x2 = self.x1 + self.l2 * np.sin(self.theta2)
        self.y2 = self.y1 - self.l2 * np.cos(self.theta2)

        self.trail.append([self.x2, self.y2])
        if len(self.trail) > self.trail_length:
            self.trail.pop(0)

    def compute_acceleration(self):
        delta = self.theta1 - self.theta2
        denom = 2 * self.m1 + self.m2 - self.m2 * np.cos(2 * delta)

        theta1_acc_num = (
                -self.g * (2 * self.m1 + self.m2) * np.sin(self.theta1)
                - self.m2 * self.g * np.sin(self.theta1 - 2 * self.theta2)
                - 2 * np.sin(delta) * self.m2 * (
                        self.theta2_dot ** 2 * self.l2 +
                        self.theta1_dot ** 2 * self.l1 * np.cos(delta)
                )
        )
        theta1_acc = theta1_acc_num / (self.l1 * denom)

        theta2_acc_num = (
                2 * np.sin(delta) * (
                self.theta1_dot ** 2 * self.l1 * (self.m1 + self.m2) +
                self.g * (self.m1 + self.m2) * np.cos(self.theta1) +
                self.theta2_dot ** 2 * self.l2 * self.m2 * np.cos(delta)
        )
        )
        theta2_acc = theta2_acc_num / (self.l2 * denom)

        return theta1_acc, theta2_acc

    def update(self, dt):
        theta1_acc, theta2_acc = self.compute_acceleration()

        self.theta1_dot += theta1_acc * dt
        self.theta2_dot += theta2_acc * dt

        self.theta1_dot = np.clip(self.theta1_dot, -self.max_angular_velocity, self.max_angular_velocity)
        self.theta2_dot = np.clip(self.theta2_dot, -self.max_angular_velocity, self.max_angular_velocity)

        self.theta1 += self.theta1_dot * dt
        self.theta2 += self.theta2_dot * dt
        self.update_positions()

    def draw(self):
        self.rod1.set_data([0, self.x1], [0, self.y1])
        self.rod2.set_data([self.x1, self.x2], [self.y1, self.y2])
        self.mass1.set_offsets([[self.x1, self.y1]])
        self.mass2.set_offsets([[self.x2, self.y2]])

        if self.trail:
            trail_positions = np.array(self.trail)
            alphas = np.linspace(0.1, 0.7, len(self.trail))  # Fading effect
            self.trail_scatter.set_offsets(trail_positions)
            self.trail_scatter.set_alpha(alphas)
            self.trail_scatter.set_sizes([self.trail_size] * len(self.trail))

        return [self.rod1, self.rod2, self.mass1, self.mass2, self.trail_scatter]

    @staticmethod
    def run():
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True)

        pendulums = [
            DoublePendulum(ax, theta1=np.pi / 1.5 + 0.3, color='red'),
            DoublePendulum(ax, theta1=np.pi / 2 - 0.2, color='blue', m1=1.5),
            DoublePendulum(ax, l1=0.8, l2=1.2, color='green', theta2=np.pi / 2 + 0.15)
        ]

        dt = 0.033
        interval = 33

        def animate(frame):
            artists = []
            for p in pendulums:
                p.update(dt)
                artists.extend(p.draw())
            return artists

        ani = FuncAnimation(fig, animate, interval=interval, blit=True, save_count=1000)
        plt.title("Double Pendulum Animation")
        ani.save('results/double_pendulum.gif', writer='imagemagick', fps=30, dpi=80)
        plt.show()
