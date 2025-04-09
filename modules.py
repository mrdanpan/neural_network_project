import numpy as np

# Abstract Module
class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass
    
# Linear Module
class Linear(Module):
    def __init__(self, _input, _output, seed = None):
        self._input = _input
        self._output = _output
        if seed is not None: np.random.seed(seed)
        self._parameters = np.random.rand(_input, _output)

    def zero_grad(self):
        self._gradient = np.zeros(self._parameters.shape)
    
    def forward(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1) # treat as batch array with only one batch
        assert X.shape[1] == self._input

        # out = matrix multiplication with parameters
        return X @ self._parameters

    def backward_update_gradient(self, inp, delta):
        # check input shape
        if len(inp.shape) == 1:
            inp = inp.reshape(1, -1) # treat as batch array with only one batch
        assert inp.shape[1] == self._input

        # check delta shape
        if len(delta.shape) == 1:
            delta = delta.reshape(-1, 1)
            
        assert delta.shape[1] == self._output, f'delta.shape[1] = {delta[1].shape}, self._output = {self._output}'

        # accumulate gradient: loop over batch
        for b, d in zip(inp, delta):
            self._gradient += b.reshape(-1, 1) @ d

    def update_parameters(self, gradient_step=0.001):
        self._parameters -= gradient_step*self._gradient

    def backward_delta(self, inp, delta):
        ## Calcul la derivee de l'erreur

        # check delta shape
        if len(delta.shape) == 1:
            delta = delta.reshape(1, -1)
        assert delta.shape[1] == self._output
        
        # in the linear case: deltas don't depend on input
        return delta @ self._parameters
