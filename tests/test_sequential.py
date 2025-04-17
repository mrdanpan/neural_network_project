
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
import torch

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
        # print(f"Grad Lin1: {model.modules[0]._gradient}")
        # print(f"Grad Lin2: {model.modules[2]._gradient}")
        # update training metrics 
        all_losses.append(epoch_loss)
        
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

def score_tensor(X_test, y_test, model):
    num_correct = 0
    for X, y in zip(X_test, y_test):
        X = model.forward(X)
        y_pred = 0 if X.item() < 0.5 else 1
        if y_pred == y.item(): 
            num_correct += 1
    
    return num_correct / len(X_test)

# Hyperparams 
seed = 10
lr = 0.001
batch_size = 200
# Data preparation
c1 = [1,2]; c2 = [3,5]
X, y = prepare_binary_class_data(c1, c2, n = 100, normalize=True, seed = seed)
X_train, y_train, X_test, y_test = split_train_test_data(X, y, perc_test=.2, seed = seed)
plot_data(X,y)

model_1 = Sequential([Linear(2, 4, seed = seed), TanH(), Linear(4, 1, seed = seed), Sigmoid()])
model_2 = Sequential([Linear(2, 4, seed = seed), TanH(), Linear(4, 1, seed = seed), Sigmoid()])
# Later, for torch params
W1_np = model_2._parameters[0].copy()
W2_np = model_2._parameters[2].copy()

mse_class = MSELoss()
all_losses_1 = train_model(X_train, y_train, model_1, loss_class=mse_class, learning_rate = lr, batch_size=batch_size, nb_epochs=1, seed = seed, verbose = False)
print(f'Score for model with {1} training epochs: {score(X_test, y_test, model_1)} ')

n_epochs = 1000
print(f"Params before training model (for {n_epochs} epochs)")
print("Lin1: ", model_2._parameters[0])
print("Lin2: ", model_2._parameters[2])
all_losses_100 = train_model(X_train, y_train, model_2, loss_class=mse_class, learning_rate = lr, batch_size=1, nb_epochs=n_epochs, seed = seed, verbose = False)
print(f'Score for model with {n_epochs} training epochs: {score(X_test, y_test, model_2)} ')
print("Params after training model (for 100 epochs)")
print("Lin1: ", model_2._parameters[0])
print("Lin2: ", model_2._parameters[2])

# plot_loss(all_losses_100,  title = "Losses for model trained on 100 epochs")
# plt.show()

# COMPARE TO TORCH...
def train(X_train, y_train, batch_size, model, loss_fn, optim, nb_epochs, seed = 10):
    model.train(True)
    all_losses = []
    
    for epoch in range(nb_epochs):
        model.zero_grad()
        if seed is not None:
            torch.manual_seed(seed)
            
        permutation = list(range(len(X_train)))
        np.random.shuffle(permutation)

        epoch_loss = 0
        
        for i in range(0, len(permutation), batch_size):
            model.zero_grad()
            X, y = X_train[permutation[i: i + batch_size]], y_train[permutation[i: i + batch_size]]
            
            # forward
            y_pred = model.forward(X)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss

            # backward
            optim.zero_grad()
            loss.backward()

            optim.step()
            
        # update training metrics 
        all_losses.append(epoch_loss.item())
    return all_losses

seed = 10
lr = 0.001
batch_size = 200
torch.manual_seed(seed)

net = torch.nn.Sequential(
    torch.nn.Linear(2,4, dtype=torch.float64, bias = False),
    torch.nn.Tanh(),
    torch.nn.Linear(4,1, dtype=torch.float64, bias = False),
    torch.nn.Sigmoid()
) 
with torch.no_grad():
    net[0].weight.copy_(torch.tensor(W1_np.T, dtype=torch.float64))
    net[2].weight.copy_(torch.tensor(W2_np.T, dtype=torch.float64))
    
loss = torch.nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr = lr)

X_train = torch.from_numpy(X_train); y_train = torch.from_numpy(y_train).to(torch.float64)
X_test = torch.from_numpy(X_test); y_test = torch.from_numpy(y_test).to(torch.float64)

print()
print("TORCH MODEL:")

all_losses = train(X_train, y_train, batch_size, net, loss_fn = loss, optim = optim, nb_epochs = 1, seed = 10)
print(f'Score for model with 1 training epoch: {score_tensor(X_test, y_test, net)} ')

net2 = torch.nn.Sequential(
    torch.nn.Linear(2,4, dtype=torch.float64, bias = False),
    torch.nn.Tanh(),
    torch.nn.Linear(4,1, dtype=torch.float64, bias = False),
    torch.nn.Sigmoid()
) 
loss = torch.nn.MSELoss()
optim = torch.optim.SGD(net2.parameters(), lr = lr)

with torch.no_grad():
    net2[0].weight.copy_(torch.tensor(W1_np.T, dtype=torch.float64))
    net2[2].weight.copy_(torch.tensor(W2_np.T, dtype=torch.float64))
    

print(f"Params before training model (for {n_epochs} epochs)")
for o1, o2 in zip(net2[0].named_parameters(), net2[2].named_parameters()):
    print("Lin1: ", o1[1])
    print("Lin2: ", o2[1])
    
all_losses = train(X_train, y_train, batch_size, net2, loss_fn = loss, optim = optim, nb_epochs = n_epochs, seed = 10)
print(f'Score for model with {n_epochs} training epochs: {score_tensor(X_test, y_test, net2)} ')
print(f"Params after training model (for {n_epochs} epochs)")
for o1, o2 in zip(net2[0].named_parameters(), net2[2].named_parameters()):
    print("Lin1: ", o1[1])
    print("Lin2: ", o2[1])

# Weights are slightly different. But do not dismay dear reader! For this is likely small changes accumulated 
# over time. If you look at the weights for one epoch, they are the same! (Our module holds to the eighth
# decimal place or so, whereas tensors to the fourth.)