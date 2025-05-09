import pickle
import numpy as np
import path_config as path_config
from neural_network.modules import *
from neural_network.losses import *
from neural_network.optim import *
from neural_network.train_gen import *
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tests.test_autoenc_latent import *

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

X, Y = zip(*[small_dataset[i] for i in range(len(small_dataset))])
X_np = np.array([x.numpy().flatten().astype(np.float32) for x in X])  
y_np = np.array([y for y in Y])

train_idx, test_idx = get_split_indices(X, perc_test=0.2, seed=10)
X_train_np, X_test_np = X_np[train_idx], X_np[test_idx]
_, y_test_np = y_np[train_idx], y_np[test_idx]

weights_path = 'tests/autoencoder_results/autoencoder_params_MSE_epoch2000.pkl'
with open(weights_path, 'rb') as f:
    weights = pickle.load(f)

num_pixels = np.matrix.flatten(X_train_np[0]).shape[0] # 786
hidden1_num_neurons = 256
hidden2_num_neurons = 2
encoder = Sequential([
                    # Encoder
                    Linear(num_pixels, hidden1_num_neurons, weight_initialisation="He"), TanH(), 
                    Linear(hidden1_num_neurons, hidden2_num_neurons, weight_initialisation="He")
                        ])
for i, layer in enumerate(encoder.modules):
    layer._parameters = weights[i]

# Visualize each type of clothing
labels = {0:"t-shirt", 1: "trouser", 2: "pullover", 3:"dress", 4:"coat", 5:"sandal",
          6:"shirt", 7:"sneaker", 8:"bag", 9:"ankle boot"}
unique_samples = {}
for x, y in zip(X_np, y_np):
    if y not in unique_samples:
        unique_samples[y] = x
    if len(unique_samples) == 10:
        break
# Convert to a list to display or use
X_one_per_class = [unique_samples[i] for i in sorted(unique_samples)]
plt.figure(figsize=(12,3))
for i in range(len(unique_samples.keys())):
    plt.subplot(1,10, i+1)
    plt.imshow(X_one_per_class[i].reshape(28,28))
    plt.xticks([]); plt.yticks([])
    plt.title(f'{labels[i]}')
plt.savefig(autoenc_dir + "fashion_items.png")
    
# Plot 2D representation of latent space 
autoenc_dir = 'tests/autoencoder_results/'
latent_preds = encoder.forward(X_test_np)
label_groups = {
    "t-shirt": 0, "shirt": 0, "pullover": 0,        # tops
    "dress": 1, "coat": 1,                          # over-clothing
    "trouser": 2,                                   # bottoms
    "sandal": 3, "sneaker": 3, "ankle boot": 3,     # footwear
    "bag": 4                                        # accessories
}
group_colors = {i: cm.Set2(i / 5) for i in range(5)} 

preds_and_labels = list(zip(latent_preds, y_test_np))

plt.figure()
seen = set()

for latent, label_idx in preds_and_labels:
    label_idx = int(label_idx)
    label_name = labels[label_idx]
    group_id = label_groups[label_name]
    color = group_colors[group_id]

    if label_name not in seen:
        plt.plot(latent[0], latent[1], '.', label=label_name, color=color)
        seen.add(label_name)
    else:
        plt.plot(latent[0], latent[1], '.', color=color)

plt.legend()
plt.title("2D latent space by class")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.savefig(autoenc_dir + "2d_latent_space.png")