import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def tanh(X):
    return np.tanh(X)

def softmax(X):
    D = -np.max(X, axis = 1).reshape(-1,1) # Coeff to help with stability of softmax
    denom = np.sum(np.exp(X + D), axis = 1)
    return (np.exp(X.T + D.T)/ denom).T

def log_softmax(X): # Note, this is natural log
    return np.log(softmax(X))

def jacob_softmax(S):
    """
    Refer to: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    
    Returns the derivative of Softmax(X) = S (s1, ..., s_dim) in terms of its 
    inputs X (x1, ..., x_dim). 
    dS/dX = jacob(S) of shape (j, j), where jacob(S)ij = DiSj = d(Sj)/d(xi) 
          = Sj(1 - Si) if i = j, -Sj * Si otherwise
          
    Args:
        S (np.ndarray): (batchsize, num_classes)
    Returns:
        res (np.ndarray): jacobian matrix (batchsize, num_classes, num_classes)
    """
    res = np.zeros(shape=(S.shape[0],S.shape[1],S.shape[1]))
    for i, row in enumerate(S):
        # compute jacobian for current batch example
        jacob_i = np.diagflat(row) - np.outer(row, row) 
        res[i] = jacob_i 
    return res
