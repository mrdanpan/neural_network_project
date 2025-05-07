import path_config as path_config
from neural_network.modules import *
from neural_network.losses import *
from neural_network.optim import *
from neural_network.train_gen import *
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision.datasets import Flowers102, Caltech101
from torchvision import transforms
import pickle
import os

autoenc_dir = 'tests/autoencoder_results/'
os.makedirs(autoenc_dir, exist_ok=True)

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
    transforms.Grayscale(),
    transforms.ToTensor()           
])
dataset = Flowers102(root='./data', download=True, transform=custom_transform)
small_dataset = Subset(dataset, range(1000))

X, _ = zip(*[small_dataset[i] for i in range(len(small_dataset))])
X = np.array([np.matrix.flatten(x.numpy().astype(np.float64)) for x in X])

X_train, X_test = split_train_test_data(X, perc_test = 0.2, seed = 10)

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(transforms.ToPILImage()(X_train[i]))
im_f = autoenc_dir + "dataset_images"
plt.savefig(im_f)

num_pixels = np.matrix.flatten(X_train[0]).shape[0] # 65356
hidden1_num_neurons = 1000
hidden2_num_neurons = 100
autoencoder = Sequential([Linear(num_pixels, hidden1_num_neurons), TanH(), Linear(hidden1_num_neurons,hidden2_num_neurons), TanH(),
                        Linear(hidden2_num_neurons, hidden1_num_neurons), TanH(), Linear(hidden1_num_neurons,num_pixels), Sigmoid()])


loss = MSELoss()
optim = Optim(autoencoder, loss, eps=1e-2)
all_losses = MBGD(X_train, X_train, autoencoder, loss, optim, batch_size = 10, nb_epochs = 2, seed = None, verbose = True, save_params = True)
params = autoencoder._parameters
for i in [0,-1]:
    with open(f'{autoenc_dir}/autoencoder_params_MSE_epoch{i}.pkl', 'wb') as f:
        pickle.dump(params[i], f)
    
plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(2,10, i*2 + 1)
    plt.imshow(X_test[i].reshape(256,256), cmap = 'grey')
    plt.xticks([]); plt.yticks([])
    plt.title(f'input {i}')

    plt.subplot(2,10, i*2 + 2)
    image_pred = autoencoder.forward(X_test[i]).reshape(256,256)
    plt.title(f'output {i}')
    plt.xticks([]); plt.yticks([])
    plt.imshow(image_pred, cmap='grey')
plt.savefig(f'{autoenc_dir}model_preds.jpg')
