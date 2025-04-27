from src.tasks import *
from src.utils.neural_helpers import Runnable


def main():
    try:
        while True:
            scripts: dict[str, type[Runnable]] = {}

            def add_runnable_class(name: str, cls: type[Runnable]):
                if not issubclass(cls, Runnable):
                    raise TypeError(f"Class {cls.__name__} does not inherit from Runnable")
                scripts[name] = cls

            add_runnable_class("1", Perceptron)
            add_runnable_class("2", MLP)
            add_runnable_class("3", HopfieldNetwork)
            add_runnable_class("4", QLearningAgent)
            # add_runnable_class("5": PoleBalancing) # optional
            add_runnable_class("6", LSystem)
            add_runnable_class("7", IFS)
            add_runnable_class("8", MandelbrotSet)
            add_runnable_class("9", FractalTerrain)
            # add_runnable_class("10", LogisticsChaos)
            # add_runnable_class("11", DoublePendulum) # optional
            # add_runnable_class("12", CellularAutomata)

            print("\n🚀 Select task:")
            for name, cls in scripts.items():
                print(f" {name} - {cls.__doc__.strip()}")
            print(" 0 - Exit")

            choice = input("Enter your choice: ")

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
