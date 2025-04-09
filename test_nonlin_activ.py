import numpy as np
import matplotlib.pyplot as plt

# MLP: Linear -> TanH -> Linear -> Sigmoid
# Classification: 0 or 1 

# Initialise data
def prepare_binary_class_data(c1, c2, v1 = 0, v2 = 1, n = 100, normalize = True, seed = 10):
    """Creates binary classification data X and Y, where X ∈ R^n and Y ∈ {v1, v2},
    where v1 is typically 0 or -1, and v2 is typically 1. Data is constructed by 
    Args:
        c1 (tuple): _description_
        c2 (tuple): _description_
        n (int): _description_
        v1 (int, optional): Class of points centered at c1. Defaults to 0.
        v2 (int, optional): Class of points centered at c2. Defaults to 1.
        normalize (bool, optional): Normalizes the data. Defaults to True.
        seed (int, optional): Randomizer seed. Defaults to 10.
    """
    assert (v1 in [-1,0]) and (v2 in [1]), f"Expected v1 in [-1, 0], v2 in [1], got v1 = {v1}, v2 = {v2}"
    rng = np.random.default_rng(seed = seed) if seed is not None else np.random.default_rng()
    
    X_c1 = (np.array([c1 for i in range(n)]).T + rng.normal(size = (len(c1), n))).T
    X_c2 = (np.array([c2 for i in range(n)]).T + rng.normal(size = (len(c2), n))).T
    y_c1 = np.array(n * [v1]); y_c2 = np.array(n * [v2])
    
    X = np.concatenate((X_c1, X_c2))
    y = np.concatenate((y_c1, y_c2)).reshape(-1, 1)
    
    if normalize:
        X = (X - np.mean(X)) / np.std(X)
        y = (y - np.mean(y)) / np.std(y)        
    return X, y
    
c1 = [1,1]; c2 = [3,3]
X, y = prepare_binary_class_data(c1, c2, n = 100, normalize=False)

