from abc import abstractmethod

import numpy as np


class Runnable:
    @staticmethod
    @abstractmethod
    def run():
        pass


class Plot2DBoundary:
    @abstractmethod
    def predict(self, X: np.ndarray):
        pass
