
import path_config
import numpy as np
import matplotlib.pyplot as plt
from neural_network.modules import * 
from neural_network.losses import *
from test_nonlin_activ import (
    prepare_binary_class_data, 
    split_train_test_data, 
    plot_loss,
    plot_data
)


def train_model(X_train, y_train, model, loss_class, learning_rate, batch_size = 1, nb_epochs = 100, seed = None, verbose = True):
    all_losses = []
    
    for epoch in range(nb_epochs):
        model.zero_grad()
        if seed is not None:
            np.random.seed(seed + epoch)
            
        permutation = list(range(len(X_train)))
        np.random.shuffle(permutation)

        epoch_loss = 0
        
        for i in range(0, len(permutation), batch_size):
            model.zero_grad()
            X, y = X_train[permutation[i: i + batch_size]], y_train[permutation[i: i + batch_size]]

            # forward
            y_pred = model.forward(X)
            loss = loss_class.forward(y, y_pred)
            epoch_loss += loss

            # backward
            delta_loss = loss_class.backward(y, y_pred)
            model.backward_update_gradient(X, delta_loss)
            model.update_parameters(learning_rate)

        # update training metrics 
        all_losses.append(epoch_loss[0])
        
        if verbose: 
            print(f"Epoch {epoch} Loss: {epoch_loss}")

    return all_losses

def score(X_test, y_test, model):
    num_correct = 0
    for X, y in zip(X_test, y_test):
        # Run the data through the modules
        X = model.forward(X)
        # Calculate score
        y_pred = 0 if X.item() < 0.5 else 1 
        if y_pred == y: num_correct += 1 
    return num_correct / len(X_test)

# Hyperparams 
seed = 10
lr = 0.0001
batch_size = 5

# Data preparation
c1 = [1,1]; c2 = [3,3]
X, y = prepare_binary_class_data(c1, c2, n = 100, normalize=True)
X_train, y_train, X_test, y_test = split_train_test_data(X, y, perc_test=.2, seed = seed)


model_1 = Sequential([Linear(2, 4), TanH(), Linear(4, 1), Sigmoid()])
model_2 = Sequential([Linear(2, 4), TanH(), Linear(4, 1), Sigmoid()])
mse_class = MSELoss()

all_losses_1 = train_model(X_train, y_train, model_1, loss_class=mse_class, learning_rate = lr, batch_size=1, nb_epochs=1, seed = seed, verbose = False)
print(f'Score for model with 1 training epoch: {score(X_test, y_test, model_1)} ')

all_losses_100 = train_model(X_train, y_train, model_2, loss_class=mse_class, learning_rate = lr, batch_size=1, nb_epochs=100, seed = seed, verbose = False)
print(f'Score for model with 100 training epochs: {score(X_test, y_test, model_2)} ')

plot_loss(all_losses_100,  title = "Losses for model trained on 100 epochs")
plt.show()