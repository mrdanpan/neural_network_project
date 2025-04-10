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

class MSELoss(Loss):
    def forward(self, y, yhat):
        """Returns the Mean Standard Error Loss between the targets (y) and the predictions (yhat)
        MSE = || y - yhat || ** 2 

        Args:
            y (numpy array): targets of shape (batchsize, d)
            yhat (numpy array): predictions of model of shape (batchsize, d)

        Returns:
            loss: mean squared error of predictions vs targets, of shape (batchsize,)
        """        
        # Assertions
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(yhat.shape) == 1:
            yhat = yhat.reshape(-1, 1)

        # Compute loss along second axis (one loss per example)
        return np.linalg.norm(y-yhat, axis=1) ** 2 

    def backward(self, y, yhat):
        """Calculates the gradient of MSE loss in terms of yhat
        dMSE/dyhat = -2 * ||y - yhat||

        Args:
            y (numpy array): targets of shape (batchsize, d)
            yhat (numpy array): predictions of model of shape (batchsize, d)
        """
        # Assertions
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(yhat.shape) == 1:
            yhat = yhat.reshape(-1, 1)

        # Compute gradient
        grad = -2 * np.linalg.norm(y-yhat, axis=1)
        return grad 
    