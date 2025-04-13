import path_config
import numpy as np
import matplotlib.pyplot as plt
from neural_network.modules import * 
from neural_network.losses import *

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
    return X, y
    
def split_train_test_data(X, y, perc_test = 0.2, seed = 10):
    if seed: np.random.seed(seed=seed)
    permutation = list(range(len(X)))
    np.random.shuffle(permutation)
    
    cutoff_idx = int(len(permutation)*perc_test)
    X_train, y_train = X[permutation[:cutoff_idx]], y[permutation[:cutoff_idx]]
    X_test, y_test = X[permutation[cutoff_idx:]], y[permutation[cutoff_idx:]]
    
    return X_train, y_train, X_test, y_test

def plot_data(X,y):
    colors = [('green', i) if i == 0 else ('red', i) for i in y.flatten()]
    for i in range(len(X.T[0])):  
        plt.scatter(X.T[0][i], X.T[1][i], c=colors[i][0], label = f'Class {colors[i][1]}')
    # Make sure only one label per class
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Display stuff
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title("Data points")
    plt.show()
    
    
def train_model(X_train, y_train, nb_epochs, batch_size, lr, seed = 10):

    # Create MLP: Linear -> TanH -> Linear -> Sigmoid
    lin1 = Linear(2, 4, seed = seed)
    tan = TanH()
    lin2 = Linear(4, 1, seed = seed)
    sig = Sigmoid()
    model = [lin1, tan, lin2, sig]
    
    # Loss function
    mse = MSELoss()

    ## TRAINING
    all_losses = []

    for epoch in range(nb_epochs):
        lin1.zero_grad(); tan.zero_grad(); lin2.zero_grad(); sig.zero_grad()
        if seed is not None:
            np.random.seed(seed + epoch)
        permutation = list(range(len(X_train)))
        np.random.shuffle(permutation)

        epoch_loss = 0
    
        # print(lin1._parameters)
        
        for i in range(0, len(permutation), batch_size):
            lin1.zero_grad(); tan.zero_grad(); lin2.zero_grad(); sig.zero_grad()
            X, y = X_train[permutation[i: i + batch_size]], y_train[permutation[i: i + batch_size]]

            # forward
            z1 = lin1.forward(X)
            z2 = tan.forward(z1)
            z3 = lin2.forward(z2)
            y_pred = sig.forward(z3)
            # print(f'X.shape = {X.shape}, out1.shape = {z1.shape}, out2.shape = {z2.shape}, out3.shape = {z3.shape}, ypred.shape = {y_pred.shape}')
            
            # loss
            loss = mse.forward(y, y_pred)
            epoch_loss += loss

            # backward on loss
            delta_loss = mse.backward(y, y_pred)
            
            # gradient calc and delta calc per layer (reverse order...)
            delta3 = sig.backward_delta(z3, delta_loss)
            
            lin2.backward_update_gradient(z2, delta3)
            delta2 = lin2.backward_delta(z2, delta3)

            delta1 = tan.backward_delta(z1, delta2)

            lin1.backward_update_gradient(X, delta1)
            
            # modules.backward_update_gradient(X, delta_loss)
            # no need to call lin_module.backward_delta()
            lin1.update_parameters(lr)
            tan.update_parameters(lr)
            lin2.update_parameters(lr)
            sig.update_parameters(lr)
            
        # append metric stats 
        all_losses.append(np.mean(epoch_loss).item())

    return all_losses, model 

def score(X_test, y_test, model):
    num_correct = 0
    for X, y in zip(X_test, y_test):
        # Run the data through the modules
        for module in model:
            X = module.forward(X)
        # Calculate score
        y_pred = 0 if X.item() < 0.5 else 1 
        if y_pred == y: num_correct += 1 
    return num_correct / len(X_test)


def plot_loss(losses, title = "MSE Loss"):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")


if __name__ == "__main__":
    # Hyperparams 
    seed = 10
    lr = 0.001
    batch_size = 1

    # Data preparation
    c1 = [1,1]; c2 = [3,3]
    X, y = prepare_binary_class_data(c1, c2, n = 100, normalize=True)
    X_train, y_train, X_test, y_test = split_train_test_data(X, y, perc_test=.2, seed = seed)

    all_losses_1, model_1 = train_model(X_train, y_train, nb_epochs=1, batch_size=1, lr = lr, seed = seed)
    print(f'Score for model with 1 training epoch: {score(X_test, y_test, model_1)} ')

    all_losses_100, model_100 = train_model(X_train, y_train, nb_epochs=100, batch_size=1, lr = lr, seed = seed)
    print(f'Score for model with 100 training epochs: {score(X_test, y_test, model_100)} ')

    plot_loss(all_losses_100,  title = "Losses for model trained on 100 epochs")
    plt.show()

    # Note: This is not a good model. Training doesn't increase score very much...
    # Could be due to use of MSE loss instead of typical BCE? Should probably look into this...
