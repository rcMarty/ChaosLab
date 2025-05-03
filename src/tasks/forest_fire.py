import enum

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.utils.neural_helpers import Runnable
from src.utils.visualization import save_plot
from matplotlib.animation import PillowWriter


class TreeState(enum.Enum):
    EMPTY = 0
    TREE = 1
    BURNING = 2


class ForestFireCA(Runnable):
    """
    Cellular Automaton implementing the Forest Fire model.
    """

    def __init__(self, size: int = 100, p_tree: float = 0.6, f: float = 0.001, p: float = 0.01, steps: int = 200):
        self.size = size
        self.p_tree = p_tree  # initial tree density
        self.f = f  # probability of lightning
        self.p = p  # probability of tree growth
        self.steps = steps
        self.grid = self.initialize_grid()

    def initialize_grid(self) -> np.ndarray:
        """
        Initialize grid with trees and empty cells.
        :return: numpy array representing the initial state of the grid
        """
        return np.random.choice([TreeState.EMPTY, TreeState.TREE], size=(self.size, self.size), p=[1 - self.p_tree, self.p_tree])

    def step(self, grid: np.ndarray) -> np.ndarray:
        """
        Perform one time step of the automaton.
            States:
            0 = empty, 1 = tree, 2 = burning
        :param grid: current state of the grid
        :return: new state of the grid
        """
        new_grid = grid.copy()
        for i in range(self.size):
            for j in range(self.size):
                if grid[i, j] == TreeState.BURNING:
                    new_grid[i, j] = TreeState.EMPTY  # burning becomes empty
                elif grid[i, j] == TreeState.TREE:
                    # Check neighbors for fire
                    neighbors = grid[max(i - 1, 0):min(i + 2, self.size), max(j - 1, 0):min(j + 2, self.size)]
                    if TreeState.BURNING in neighbors:
                        new_grid[i, j] = TreeState.BURNING  # tree catches fire
                    elif np.random.rand() < self.f:
                        new_grid[i, j] = TreeState.BURNING  # lightning
                elif grid[i, j] == TreeState.EMPTY:
                    if np.random.rand() < self.p:
                        new_grid[i, j] = TreeState.TREE  # new tree grows
        return new_grid

    def simulate(self) -> list[np.ndarray]:
        """
        Simulate the automaton over time.
        :return: list of grid states at each time step
        """
        history = [self.grid.copy()]
        for _ in range(self.steps):
            self.grid = self.step(self.grid)
            history.append(self.grid.copy())
        return history

    @staticmethod
    def run():
        model = ForestFireCA(size=100, steps=200)
        frames = model.simulate()
        frames = [np.array([[{TreeState.EMPTY: 1, TreeState.TREE: 2, TreeState.BURNING: 0}[cell] for cell in row] for row in frame]) for frame in frames]

        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.cm.get_cmap("RdYlGn", 3)
        im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=2)
        plt.title("Forest Fire Simulation")

        def update(frame):
            im.set_array(frame)
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
        plt.show()

        # Save final frame
        save_plot(fig, "forest_fire_final")
        # Save animation
        ani.save("results/forest_fire_simulation.gif", writer=PillowWriter(fps=10))
