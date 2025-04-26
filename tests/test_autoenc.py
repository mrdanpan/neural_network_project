import path_config as path_config
from neural_network.modules import *
from neural_network.losses import *
from neural_network.optim import *
from neural_network.train_gen import *
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import Flowers102
from torchvision import transforms

# Data preparation
def split_train_test_data(X, perc_test = 0.2, seed = 10):
    if seed: np.random.seed(seed=seed)
    permutation = list(range(len(X)))
    np.random.shuffle(permutation)
    
    cutoff_idx = int(len(permutation)*(1 - perc_test))
    X_train = X[permutation[:cutoff_idx]]
    X_test = X[permutation[cutoff_idx:]]
    
    return X_train, X_test

custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()           
])
X, _ = zip(*Flowers102(root='./data',download=True, transform=custom_transform))
X = np.array([x.numpy() for x in X])

X_train, X_test = split_train_test_data(X, perc_test = 0.2, seed = 10)


# Display images: we are working with Oxford Flowers 102 dataset
verbose = False
if verbose: 
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(transforms.ToPILImage()(X_train[i]))
    plt.show()

## EXPERIMENTING ON AUTOENCODER -> number of neurons for the compression...
hidden1_num_neurons = 100
hidden2_num_neurons = 10
autoencoder = Sequential([Linear(256, hidden1_num_neurons), TanH(), Linear(hidden1_num_neurons,hidden2_num_neurons), TanH(),
                        Linear(hidden2_num_neurons, hidden1_num_neurons), TanH(), Linear(hidden1_num_neurons,256), Sigmoid()])

loss = CrossEntropy()
optim = Optim(autoencoder, loss)

all_losses = MBGD(X_train, X_train, autoencoder, loss, optim, batch_size = 10, nb_epochs = 100, seed = None, verbose = True, save_params = False)
# TODO: flatten images to vectors, also maybe decide on smaller pixel images.... this one leads to a lot of neurons...