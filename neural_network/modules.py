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
        # accumulate gradient
        self._gradient += inp.T @ delta

    def update_parameters(self, gradient_step=0.001):
        self._parameters -= gradient_step*self._gradient

    def backward_delta(self, inp, delta):
        ## Calcul la derivee de l'erreur

        # check delta shape
        if len(delta.shape) == 1:
            delta = delta.reshape(-1, 1)
        assert delta.shape[1] == self._output
        
        # in the linear case: deltas don't depend on input
        return delta @ self._parameters.T
    
# Sequential Module
class Sequential(Module):
    def __init__(self, modules: list[Module]):
        self.modules = modules
        self._parameters = [module._parameters for module in self.modules]
    
    def zero_grad(self):
        ## Annule gradient
        for module in self.modules:
            module.zero_grad()

    def forward(self, X):
        ## Calcule la passe forward
        out = X
        for module in self.modules:
            out = module.forward(out)
        return out

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        for module in self.modules:
            module.update_parameters(gradient_step)

    def backward_update_gradient(self, inp, delta):
        ## Met a jour la valeur du gradient
        self.forward_delta(0, inp, delta, update_grad=True)

    def backward_delta(self, inp, delta):
        return self.forward_delta(0, inp, delta, update_grad=False)

    def forward_delta(self, i_module, inp, delta, update_grad=False):
        if i_module >= len(self.modules):
            return delta
        current_module = self.modules[i_module]
        current_delta = self.forward_delta(i_module+1, current_module.forward(inp), delta, update_grad)
        if update_grad:
            current_module.backward_update_gradient(inp, current_delta)
        return current_module.backward_delta(inp, current_delta)
    
# Activation function modules  
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return sigmoid(X)

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        sigmoid_term = (sigmoid(input) * (1 - sigmoid(input)))
        return delta * sigmoid_term

def tanh(X):
    return (np.e ** (2*X) - 1) / (np.e ** (2*X) + 1) 

class TanH(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        # returns tanh(x)
        return tanh(X)
    
    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        tanh_term = (1 - tanh(input)**2)
        return delta * tanh_term
    
