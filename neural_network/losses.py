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

        # Compute loss 
        return np.mean((y-yhat)**2)

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
            
        batch_size, dim = y.shape

        # Compute gradient
        grad = -2 * (y - yhat) / (dim * batch_size) # Average over all elements in batch...
        return grad
    
    
class CrossEntropy(Loss):
    def forward(self, y, yhat, log = False, eps = 1e-10):
        """Returns the CE loss 

        Args:
            y (np.ndarray): array of indices of the correct class 
            yhat (np.ndarray): predictions corresponding to classes,
                            of size (batch_size, num_classes)
            log (boolean): decides whether we want logCE or not 
        """
        ce = np.zeros(shape = (y.shape[0]))
        for i, (row_y, row_yhat) in enumerate(zip(y, yhat)):
            idx_1 = np.where(row_y == 1)[0][0]
            if not log:
                ce[i] = - np.log(row_yhat[idx_1]) # since we are dealing with one-hots 
            else: 
                ce[i] = - row_yhat[idx_1]

        return ce

    def backward(self, y, yhat, log = False):
        return yhat - y if not log else np.exp(yhat) - y
    
    
