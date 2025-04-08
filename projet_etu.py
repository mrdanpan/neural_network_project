import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

    def assert_type_and_shape(self, y, yhat):
        # Asserts to make sure y and yhat are numpy arrays, and of correct shape
        assert isinstance(y, np.ndarray) and isinstance(yhat, np.ndarray), "y and yhat are not numpy arrays!"
        assert y.shape == yhat.shape, f"y and yhat are not the same shape: y: {y.shape}, yhat: {yhat.shape}"


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
    
