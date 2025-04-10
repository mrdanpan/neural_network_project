import numpy as np

x = np.random.random(size=(5,))
y = np.random.random(size=(5,1))

print(x * y.T)
