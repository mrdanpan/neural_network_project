import path_config as path_config
from neural_network.modules import *
from neural_network.losses import *
from neural_network.optim import *
from neural_network.train_gen import *
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import Flowers102
from torch.utils.data import Subset
from torchvision import transforms
import pickle

plt.figure(figsize=(5,10))
plt.show()

# # Data preparation
# def split_train_test_data(X, perc_test = 0.2, seed = 10):
#     if seed: np.random.seed(seed=seed)
#     permutation = list(range(len(X)))
#     np.random.shuffle(permutation)
    
#     cutoff_idx = int(len(permutation)*(1 - perc_test))
#     X_train = X[permutation[:cutoff_idx]]
#     X_test = X[permutation[cutoff_idx:]]
    
#     return X_train, X_test

# # Define your transform
# custom_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.Grayscale(),
#     transforms.ToTensor()
# ])

# # Load the full dataset once
# dataset = Flowers102(root='./data', download=True, transform=custom_transform)
# small_dataset = Subset(dataset, range(500))

# X, _ = zip(*[small_dataset[i] for i in range(len(small_dataset))])
# X = np.array([x.numpy().astype(np.float64).squeeze() for x in X])

# plt.figure(figsize=(8,8))
# for i in range(16):
#     print(X[i].shape)
#     plt.subplot(4,4, i+1)
#     plt.imshow(X[i], cmap = 'grey')
#     plt.xticks([]); plt.yticks([])
# plt.show()