import numpy as np
from .modules import Module
from .losses import Loss

class Optim():
    def __init__(self, net: Module, loss: Loss, eps = 10e-4):
        self.net = net
        self.loss = loss
        self.eps = eps
        
    def step(self, batch_x, batch_y):
        # Calculates the sortie of batch_x
        y_pred = self.net.forward(batch_x)
        
        # Calculates cost comparing predictions to targets (batch_y)
        loss = self.loss.forward(batch_y, y_pred)
        
        # Backwards pass
        delta_loss = self.loss.backward(batch_y, y_pred)
        
        # Update params
        self.net.backward_update_gradient(batch_x, delta_loss)
        self.net.update_parameters(gradient_step=self.eps)
        
    
        
        