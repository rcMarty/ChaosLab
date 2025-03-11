import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.utils.neural_helpers import Plot2D


def visualize_data(X: np.ndarray, y: np.ndarray = None, title: str = "2D Data", labels: tuple = ("X1", "X2"), show: bool = True) -> None:
    """
    Univerzální funkce pro vizualizaci bodových dat (2D datasetů).

    :param X: numpy array, shape (n_samples, 2) - data (např. body pro perceptron, XOR)
    :param y: numpy array, shape (n_samples,) - třídy (0/1 nebo více tříd)
    :param title: str - název grafu
    :param labels: tuple - popisky os (např. ('X1', 'X2'))
    :param show: bool - zda zobrazit graf ihned nebo vrátit fig objekt
    :return: plt.figure - pokud show=False, vrátí objekt fig
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if y is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.7)
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
    else:
        ax.scatter(X[:, 0], X[:, 1], color='black', alpha=0.5)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)

    if show:
        plt.show()
    else:
        return fig


def animate_learning(error_history: list[int], title: str = "Error Reduction Over Time") -> None:
    """
    Animates the training error over time.

    :param error_history: List of errors at each epoch
    :param title: Graph title
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(error_history))
    ax.set_ylim(0, max(error_history) + 1)
    line, = ax.plot([], [], lw=2)

    def update(frame: int) -> tuple[plt.Line2D]:
        line.set_data(range(frame), error_history[:frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(error_history), interval=50)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Error Count")
    plt.show()


def visualize_grid(grid: np.ndarray, figsize: tuple[int, int] = (6, 6), title: str = "Grid") -> None:
    """
    Vykreslí matici (např. Cellular Automata, Q-learning mapu).

    :param figsize: tuple(int,int) - velikost grafu
    :param grid: numpy array, shape (m, n) - mřížka pro vizualizaci
    :param title: str - název grafu
    """
    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.show()


def plot_2d_decision_boundary(model: Plot2D, X: np.ndarray, y: np.ndarray, title: str = "Decision Boundary") -> None:
    """Vykreslí rozhodovací hranici pro 2D dataset (např. perceptron, XOR síť).

    :param model: Perceptron - natrénovaný model
    :param X: numpy array, shape (n_samples, 2) - data
    :param y: numpy array, shape (n_samples,) - třídy
    :param title: str - název grafu
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title(title)
    plt.show()


def plot_3d_fractal(x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str = "Fractal") -> None:
    """
    Vykreslí 3D fraktál nebo chaotický systém.

    :param x: numpy array - souřadnice X
    :param y: numpy array - souřadnice Y
    :param z: numpy array - souřadnice Z
    :param title: str - název grafu
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=0.5, color="black")
    ax.set_title(title)
    plt.show()


def plot_fractal(x: np.ndarray, y: np.ndarray, title: str = "Fractal"):
    """Vykreslí fraktální obrazec.

    :param x: numpy array - souřadnice X
    :param y: numpy array - souřadnice Y
    :param title: str - název grafu
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=0.1, color="black")
    plt.title(title)
    plt.show()
