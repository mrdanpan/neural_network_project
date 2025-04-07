from projet_etu import Module
import numpy as np

class Linear(Module):
    def __init__(self, _input, _output):
        self._input = _input
        self._output = _output
        self._parameters = np.random.rand(_input, _output)

    def forward(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1) # treat as batch array with only one batch
        assert X.shape[1] == self._input, f"Shape mismatch between: input ({X.shape[1]}) and module input size ({self._input})"

        # out = matrix multiplication with parameters
        return X @ self._parameters

    def backward_update_gradient(self, inp, delta):
        
        # check input shape
        if len(inp.shape) == 1:
            inp = inp.reshape(1, -1) # treat as batch array with only one batch
        assert inp.shape[1] == self._input

        # check delta shape
        if len(delta.shape) == 1:
            delta = delta.reshape(1, -1)
        assert delta.shape[1] == self._output

        # accumulate gradient: loop over batch
        for b in inp:
            self._gradient += b.reshape(-1, 1) @ delta

    def backward_delta(self, inp, delta):
        ## Calcul la derivee de l'erreur

        # check delta shape
        if len(delta.shape) == 1:
            delta = delta.reshape(1, -1)
        assert delta.shape[1] == self._output
        
        # in the linear case: deltas don't depend on input
        return delta @ self._parameters
