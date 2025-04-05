from projet_etu import Module
import numpy as np

class Linear(Module):
    def __init__(self, _input, _output):
        self._input = _input
        self._output = _output
        self._parameters = np.random(shape=(_input, _output))

    def forward(self, X):
        assert X.shape[1] == self._input

        # out = matrix multiplication with parameters
        return X @ self._parameters