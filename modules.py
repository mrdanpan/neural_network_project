from projet_etu import Module
import numpy as np

class Linear(Module):
    def __init__(self, _input, _output):
        self._input = _input
        self._output = _output
        self._parameters = np.random.rand(_input, _output)

    def forward(self, X):
        assert X.shape[1] == self._input, f"Shape mismatch between: input ({X.shape[1]}) and number of neurons in module ({self._input})"

        # out = matrix multiplication with parameters
        return X @ self._parameters

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        self._gradient = self.backward_delta(input, delta)
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        """Calculates the cost gradient in function of the input (z) and the deltas of the 
        previous layers (delta). 
        grad = delta * dz_h/dw_h = delta * W  

        Args:
            input (np array): input to module, of shape (batchsize, d,)
            delta (np array): deltas of the following layers, of shape (batchsize, d,)
        """
        grad = delta @ self._parameters
        return grad 
