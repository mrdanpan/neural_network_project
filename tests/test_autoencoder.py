import pickle
import numpy as np
import torch
import path_config as path_config
from neural_network.modules import *
from neural_network.losses import *
from neural_network.optim import *
from neural_network.train_gen import *
from tqdm import tqdm
from torchvision.datasets import FashionMNIST
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset
import matplotlib.pyplot as plt

## % Data Preparation
autoenc_dir = 'tests/autoencoder_results/'
custom_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor()           
])
dataset = FashionMNIST(root='./data', download=True, transform=custom_transform)
small_dataset = Subset(dataset, range(10000))

# Train/test split
def get_split_indices(X, perc_test=0.2, seed=10):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - perc_test))
    return idx[:split], idx[split:]

X, _ = zip(*[small_dataset[i] for i in range(len(small_dataset))])
X_torch = torch.stack([x.flatten() for x in X])# 28x28
X_np = np.array([x.numpy().flatten().astype(np.float32) for x in X])  

train_idx, test_idx = get_split_indices(X, perc_test=0.2, seed=10)
X_train_torch, X_test_torch = X_torch[train_idx], X_torch[test_idx]
X_train_np, X_test_np = X_np[train_idx], X_np[test_idx]

# Assert equality
assert np.allclose(X_train_torch, X_train_np)
assert np.allclose(X_test_torch, X_test_np)

## % 
num_pixels = np.matrix.flatten(X_train_np[0]).shape[0] # 786
hidden1_num_neurons = 256
hidden2_num_neurons = 16
autoencoder = Sequential([
                        # Encoder
                        Linear(num_pixels, hidden1_num_neurons, weight_initialisation="He"), TanH(), 
                        Linear(hidden1_num_neurons, hidden2_num_neurons, weight_initialisation="He"), TanH(),
                        
                        # Decoder
                        Linear(hidden2_num_neurons, hidden1_num_neurons, weight_initialisation="He"), TanH(), 
                        Linear(hidden1_num_neurons,num_pixels, weight_initialisation="He"), Sigmoid()
                        ])

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256, bias = False),
            nn.Tanh(),
            nn.Linear(256, 16, bias = False),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 256, bias = False),
            nn.Tanh(),
            nn.Linear(256, 784, bias = False),
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
autoencoder_torch = Autoencoder()

weights_path = 'tests/autoencoder_results/784_256_16_lr5e-1_bs2048_ds10k_1/autoencoder_params_MSE_epoch0.pkl'
with open(weights_path, 'rb') as f:
    weights = pickle.load(f)


with torch.no_grad():
    # encoder.0
    if weights[0] is not None:
        autoencoder_torch.encoder[0].weight.copy_(torch.tensor(weights[0].T, dtype=torch.float32))
        autoencoder.modules[0]._parameters = weights[0]
        
    # encoder.2
    if weights[2] is not None:
        autoencoder_torch.encoder[2].weight.copy_(torch.tensor(weights[2].T, dtype=torch.float32))
        autoencoder.modules[2]._parameters = weights[2]

    # decoder.0
    if weights[4] is not None:
        autoencoder_torch.decoder[0].weight.copy_(torch.tensor(weights[4].T, dtype=torch.float32))
        autoencoder.modules[4]._parameters = weights[4]

    # decoder.2
    if weights[6] is not None:
        autoencoder_torch.decoder[2].weight.copy_(torch.tensor(weights[6].T, dtype=torch.float32))
        autoencoder.modules[6]._parameters = weights[6]


def compare_weights(model_weights, file_weights, layer_name):
    if file_weights is not None:
        file_weights_tensor = torch.tensor(file_weights.T, dtype=torch.float32)
        assert torch.allclose(model_weights, file_weights_tensor, atol=1e-6), f"Not the same in {layer_name}"
    else:
        print(f"Error")

# Perform comparisons
compare_weights(autoencoder_torch.encoder[0].weight.data, weights[0], "encoder[0]")
compare_weights(autoencoder_torch.encoder[2].weight.data, weights[2], "encoder[2]")
compare_weights(autoencoder_torch.decoder[0].weight.data, weights[4], "decoder[0]")
compare_weights(autoencoder_torch.decoder[2].weight.data, weights[6], "decoder[2]")

# Save dataset images
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(transforms.ToPILImage()(X_train_torch[i].reshape(28,28)))
    plt.xticks([]); plt.yticks([])
im_f = autoenc_dir + "dataset_images_torch"
plt.savefig(im_f)
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(transforms.ToPILImage()(X_test_np[i].reshape(28,28)))
    plt.xticks([]); plt.yticks([])
im_f = autoenc_dir + "dataset_images_np"
plt.savefig(im_f)

# Preliminary predictions
pred_torch = autoencoder_torch(X_test_torch[0].unsqueeze(0))
pred_np = torch.tensor(autoencoder.forward(X_test_np[0]), dtype = torch.float32)
print(torch.allclose(pred_np, pred_torch))

plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plt.imshow(pred_torch.detach().numpy().reshape(28,28))
plt.title("Torch model prediction (0 epochs)")
plt.xticks([]); plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(pred_np.reshape(28,28))
plt.title("Project model prediction (0 epochs)")
plt.xticks([]); plt.yticks([])
im_f = autoenc_dir + "torch_vs_model_preds_0epochs.png"
plt.savefig(im_f)

def plot_autoenc_preds(autoencoder, X_test, nb_epochs, is_torch = False):
    plt.figure(figsize=(12,4))
    for i in range(10):
        plt.subplot(2,10, i*2 + 1)
        plt.imshow(X_test[i].reshape(28,28))
        plt.xticks([]); plt.yticks([])
        plt.title(f'input {i}')

        plt.subplot(2,10, i*2 + 2)
        if is_torch:
            with torch.no_grad():
                input_tensor = X_test[i].clone().detach()
                image_pred = autoencoder(input_tensor).cpu().numpy()
        else:
            image_pred = autoencoder.forward(X_test[i])
        plt.title(f'output {i}')
        plt.xticks([]); plt.yticks([])
        plt.imshow(image_pred.reshape(28,28))
    if is_torch: plt.savefig(f'{autoenc_dir}model_preds_{nb_epochs}epochs_torch.jpg')
    else: plt.savefig(f'{autoenc_dir}model_preds_{nb_epochs}epochs.jpg')
    
plot_autoenc_preds(autoencoder, X_test_np, nb_epochs=0, is_torch = False)
plot_autoenc_preds(autoencoder_torch, X_test_torch, nb_epochs=0, is_torch = True)


# TRAIN MODELS
loss = MSELoss()
optim = Optim(autoencoder, loss, eps=5e-1)
n_epochs = 2000
all_losses, all_params = MBGD(X_train_np, X_train_np, autoencoder, loss, optim, batch_size = 2048, nb_epochs = n_epochs, seed=10, verbose = False, save_params = True)
# Save parameters
with open(f'{autoenc_dir}/autoencoder_params_MSE_epoch{len(all_params)}.pkl', 'wb') as f:
    pickle.dump(all_params[-1], f)

plt.figure(figsize=(8,8))
plt.plot(all_losses)
plt.title('Autoencoder Training Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig(f'{autoenc_dir}/losses.png')

plot_autoenc_preds(autoencoder, X_test_np, nb_epochs=n_epochs, is_torch=False)


def train_MBGD(model, X_train, loss_fn, optimizer, batch_size=10, nb_epochs=100, seed=10, verbose=True, save_params=False):

    all_losses = []
    all_params = [model.state_dict()] if save_params else None
    X_train = X_train.detach().numpy()

    for epoch in tqdm(range(nb_epochs)):
        model.train()
        epoch_loss = 0.0

        if seed is not None:
            np.random.seed(seed + epoch)

        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]

        dataset = TensorDataset(
            torch.tensor(X_shuffled, dtype=torch.float32),
            torch.tensor(X_shuffled, dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch) 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # epoch_loss /= len(dataloader)
        all_losses.append(epoch_loss)

        if save_params:
            all_params.append({k: v.clone().detach() for k, v in model.state_dict().items()})

        if verbose:
            print(f"Epoch {epoch + 1}/{nb_epochs} - Loss: {epoch_loss:.6f}")

    if save_params: return all_losses, all_params
    else: return all_losses

loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(autoencoder_torch.parameters(), lr=0.5)
n_epochs = 2000
all_losses, all_params = train_MBGD(autoencoder_torch, X_train_torch, loss_fn, optimizer,
                                    batch_size=2048, nb_epochs=n_epochs,
                                    seed=10, verbose=False, save_params=True)

with open(f'{autoenc_dir}/autoencoder_params_MSE_epoch{len(all_params)}_torch.pkl', 'wb') as f:
    pickle.dump(all_params[-1], f)

plt.figure(figsize=(8,8))
plt.plot(all_losses)
plt.title('Torch Autoencoder Training Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig(f'{autoenc_dir}/losses_torch_model.png')

plot_autoenc_preds(autoencoder_torch, X_test_torch, nb_epochs=n_epochs, is_torch=True)