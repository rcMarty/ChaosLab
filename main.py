from src.tasks import *
from src.utils.neural_helpers import Runnable


def main():
    try:
        while True:
            print("\n🚀 Select task:")
            print(" 1 - Perceptron")
            print(" 2 - XOR neuron network")
            print(" 3 - Hopfield network")
            print(" 4 - Q-learning (Find the Cheese)")
            print(" 5 - Fractal L-systems")
            print(" 6 - Fractal geometrics (IFS)")
            print(" 7 - Mandelbrot Set")
            print(" 8 - Logistics Chaos")
            print(" 9 - Double Pendulum")
            print("10 - Cellular Automata")
            print(" 0 - Exit")

            choice = input("Enter your choice: ")

            scripts: dict[str, type[Runnable]] = {}

            def add_runnable_class(name: str, cls: type[Runnable]):
                if not issubclass(cls, Runnable):
                    raise TypeError(f"Class {cls.__name__} does not inherit from Runnable")
                scripts[name] = cls

            add_runnable_class("1", Perceptron)
            add_runnable_class("2", MLP)
            add_runnable_class("3", HopfieldNetwork)
            # add_runnable_class("4": Qlearning)
            # add_runnable_class("5": LSystems)
            # add_runnable_class("6": IFS)
            # add_runnable_class("7": MandelbrotSet)
            # add_runnable_class("8": LogisticsChaos)
            # add_runnable_class("9": DoublePendulum)
            # add_runnable_class("10":CellularAutomata)

            if choice == "0":
                print("Exiting program ...")
                break
            elif choice in scripts:
                scripts[choice].run()
            else:
                print("Invalid choice, please try again.")
    except KeyboardInterrupt:
        print("\nCTRL+C detected\nExiting program ...")


if __name__ == "__main__":
    main()
