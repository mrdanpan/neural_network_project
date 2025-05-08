import path_config as path_config
from neural_network.modules import *
from neural_network.losses import *
from neural_network.optim import *
from neural_network.train_gen import *
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision.datasets import FashionMNIST
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
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor()           
])
dataset = FashionMNIST(root='./data', download=True, transform=custom_transform)
small_dataset = Subset(dataset, range(10000))

X, y = zip(*[small_dataset[i] for i in range(len(small_dataset))])
X = np.array([np.matrix.flatten(x.numpy().astype(np.float32)) for x in X])

X_train, X_test = split_train_test_data(X, perc_test = 0.2, seed = 10)

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(transforms.ToPILImage()(X_train[i].reshape(28,28)))
    plt.xticks([]); plt.yticks([])
im_f = autoenc_dir + "dataset_images"
plt.savefig(im_f)

def plot_autoenc_preds(autoencoder, X_test, nb_epochs):
    plt.figure(figsize=(12,4))
    for i in range(10):
        plt.subplot(2,10, i*2 + 1)
        plt.imshow(X_test[i].reshape(28,28))
        plt.xticks([]); plt.yticks([])
        plt.title(f'input {i}')

        plt.subplot(2,10, i*2 + 2)
        image_pred = autoencoder.forward(X_test[i]).reshape(28,28)
        plt.title(f'output {i}')
        plt.xticks([]); plt.yticks([])
        plt.imshow(image_pred)
    plt.savefig(f'{autoenc_dir}model_preds_{nb_epochs}epochs.jpg')

num_pixels = np.matrix.flatten(X_train[0]).shape[0] # 786
hidden1_num_neurons = 256
hidden2_num_neurons = 16
autoencoder = Sequential([
                        # Encoder
                        Linear(num_pixels, hidden1_num_neurons), TanH(), 
                        Linear(hidden1_num_neurons, hidden2_num_neurons), TanH(),
                        
                        # Decoder
                        Linear(hidden2_num_neurons, hidden1_num_neurons), TanH(), 
                        Linear(hidden1_num_neurons,num_pixels), Sigmoid()
                        ])

plot_autoenc_preds(autoencoder, X_test, nb_epochs=0)
loss = MSELoss()
optim = Optim(autoencoder, loss, eps=5e-1)
n_epochs = 2000
with open(f'{autoenc_dir}/autoencoder_params_MSE_epoch0.pkl', 'wb') as f:
        pickle.dump(autoencoder._parameters, f)
        
plt.figure()
pred = autoencoder.forward(X_test[0])
plt.imshow(pred.reshape(28,28))
plt.show()
exit()
all_losses, all_params = MBGD(X_train, X_train, autoencoder, loss, optim, batch_size = 2048, nb_epochs = n_epochs, seed = None, verbose = False, save_params = True)

# Save parameters
for i in [0, n_epochs]:
    with open(f'{autoenc_dir}/autoencoder_params_MSE_epoch{i}.pkl', 'wb') as f:
        pickle.dump(all_params[i], f)

with open(f'{autoenc_dir}/losses_training.pkl', 'wb') as f:
    pickle.dump(all_losses, f)
    
pred_test = autoencoder.forward(X_test[-1])
print(np.min(pred_test), np.max(pred_test))
print(pred_test)
    
plt.figure(figsize=(8,8))
plt.plot(all_losses)
plt.title('Autoencoder Training Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig(f'{autoenc_dir}/losses.png')
    
plot_autoenc_preds(autoencoder, X_test, nb_epochs=n_epochs)
