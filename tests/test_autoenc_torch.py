import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchvision.datasets import FashionMNIST
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import pickle

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


def train_MBGD(model, X_train, loss_fn, optimizer, batch_size=10, nb_epochs=100, seed=None, verbose=True, save_params=False):
    all_losses = []
    all_params = [model.state_dict()] if save_params else None

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_train, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(nb_epochs)):
        model.train()
        epoch_loss = 0.0

        if seed is not None:
            torch.manual_seed(seed + epoch)

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0) 

        epoch_loss /= len(dataset)
        all_losses.append(epoch_loss)

        if save_params:
            all_params.append({k: v.clone().detach() for k, v in model.state_dict().items()})

        if verbose:
            print(f"Epoch {epoch + 1}/{nb_epochs} - Loss: {epoch_loss:.6f}")

    return (all_losses, all_params) if save_params else all_losses


autoenc_dir = 'tests/autoencoder_results/'
custom_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(),
    transforms.ToTensor()           
])
dataset = FashionMNIST(root='./data', download=True, transform=custom_transform)
small_dataset = Subset(dataset, range(10000))

# Convert dataset to numpy array
X, _ = zip(*[small_dataset[i] for i in range(len(small_dataset))])
X = np.array([x.flatten() for x in X])  # 28x28

# Train/test split
def split_train_test_data(X, perc_test=0.2, seed=10):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - perc_test))
    return X[idx[:split]], X[idx[split:]]

X_train, X_test = split_train_test_data(X, perc_test=0.2, seed=10)

# Show input images
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(X_train[i].reshape(28, 28))
    plt.xticks([]); plt.yticks([])
plt.savefig(autoenc_dir + "dataset_images.png")


def plot_autoenc_preds(autoencoder, X_test, nb_epochs):
    autoencoder.eval()
    with torch.no_grad():
        plt.figure(figsize=(12, 4))
        for i in range(10):
            plt.subplot(2, 10, i*2 + 1)
            plt.imshow(X_test[i].reshape(28,28))
            plt.title(f"Input {i}")
            plt.xticks([]); plt.yticks([])

            pred = autoencoder(torch.tensor(X_test[i], dtype=torch.float32)).detach().numpy()
            plt.subplot(2, 10, i*2 + 2)
            plt.imshow(pred.reshape(28,28))
            plt.title(f"Output {i}")
            plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{autoenc_dir}/model_preds_{nb_epochs}epochs.jpg")

model = Autoencoder()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

n_epochs = 2000
plot_autoenc_preds(model, X_test, nb_epochs=0)
all_losses, all_params = train_MBGD(model, X_train, loss_fn, optimizer,
                                    batch_size=2048, nb_epochs=n_epochs,
                                    seed=None, verbose=False, save_params=True)

# Save parameters
for i, epoch_num in enumerate([0, n_epochs]):
    with open(f'{autoenc_dir}/autoencoder_params_MSE_epoch{epoch_num}.pkl', 'wb') as f:
        pickle.dump(all_params[i], f)

with open(f'{autoenc_dir}/losses_training.pkl', 'wb') as f:
    pickle.dump(all_losses, f)

# Show final prediction
model.eval()
with torch.no_grad():
    pred_test = model(torch.tensor(X_test[-1], dtype=torch.float32)).detach().numpy()
    print(np.min(pred_test), np.max(pred_test))

# Loss plot
plt.figure(figsize=(8, 8))
plt.plot(all_losses)
plt.title('Autoencoder Training Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig(f'{autoenc_dir}/losses.png')

# Prediction visualization
plot_autoenc_preds(model, X_test, nb_epochs=n_epochs)
