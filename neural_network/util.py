import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def tanh(X):
    return (np.e ** (2*X) - 1) / (np.e ** (2*X) + 1) 

def softmax(X):
    D = -np.max(X, axis = 1).reshape(-1,1) # Coeff to help with stability of softmax
    print(D)
    denom = np.sum(np.exp(X + D), axis = 1)
    return (np.exp(X.T + D.T)/ denom).T


def log_softmax(X, eps = 1e-5):
    return np.log10(softmax(X) + eps)

def deriv_softmax(X):
    """
    Refer to: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    
    Returns the derivative of Softmax(X) = S (s1, ..., s_dim) in terms of its 
    inputs X (x1, ..., x_dim). 
    dS/dX = jacob(S) of shape (j, j) = 

    Args:
        X (np.ndarray): (batchsize, dim)
    """
    pass

np.random.seed(10)
x = np.random.random(size = (5,4))
print(x)
print(softmax(x))
print(log_softmax(x))
