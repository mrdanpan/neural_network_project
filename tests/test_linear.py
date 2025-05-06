import path_config as path_config
from neural_network.modules import *
from neural_network.losses import *
import numpy as np
import matplotlib.pyplot as plt

# Train function
def train_model(X_train, y_train, model, loss_class, learning_rate, batch_size = 1, n_epochs = 100, seed = None, verbose = True):
    all_losses = []
    all_params = [model._parameters[0][0]]
    
    for epoch in range(n_epochs):
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
            # no need to call lin_module.backward_delta()
            model.update_parameters(learning_rate)

        # update parameters
        all_losses.append(np.mean(epoch_loss))
        all_params.append(model._parameters[0][0])
        
        if verbose: 
            print("current params", model._parameters)
            print(f"Epoch {epoch} Loss: {epoch_loss}")

    return all_losses, all_params

# Initialise data
def prepare_data(slope, normalize = True, seed = 10):
    np.random.seed(seed)
    X_train = np.linspace(0, 10, 50).reshape(-1, 1)
    y_train = slope * X_train + np.random.normal(size=(X_train.shape), scale=5)

    if normalize: 
        X_train = (X_train - np.mean(X_train)) / np.std(X_train)
        y_train = (y_train - np.mean(y_train)) / np.std(y_train)
    return X_train, y_train
 
if __name__ == "__main__":

    # Test different learning rates
    lrs = [float(f'10e-{i}') for i in range(3,7)]
    # lrs = [0.01, 0.005, 0.001, 0.0005, 0.0001]

    plt.figure(figsize=(12,8))
    for i,lr in enumerate(lrs):
        # Initialise hyperparams
        n_epochs = 200
        batch_size = 1
        seed = 13
        # Obtain data
        X_train, y_train = prepare_data(slope = 5, normalize = True, seed = seed)
        # Initialise model and loss 
        lin_module = Linear(1, 1, seed = seed)
        mse_loss = MSELoss()
        # Train
        all_losses, all_params = train_model(X_train, y_train, lin_module, mse_loss, lr, batch_size, n_epochs, seed = seed, verbose = False)
        
        # Visualization
        plt.subplot(len(lrs), 3, i*3 + 1)
        plt.plot(all_losses)
        plt.ylabel(f"lr = {lr}\n\nLoss")
        plt.xlabel("Epoch")
        plt.ylim(bottom = 0)
        if i == 0: plt.title("Loss over epochs")
        
        plt.subplot(len(lrs), 3, i*3 + 2)
        plt.plot(all_params)
        plt.ylabel(f"Weight")
        plt.xlabel("Epoch")
        if i == 0: plt.title("Model Param")
        
        m_hat = all_params[-1]
        predictions = m_hat * X_train
        plt.subplot(len(lrs), 3, i*3 + 3)
        plt.scatter(X_train, y_train)
        plt.plot(X_train, predictions, color = 'red', label = f'Prediction: y = {m_hat: .2f}')
        plt.legend()
        if i == 0: plt.title(f"Linear Regression against Training Data")
    plt.suptitle("Hyperparameter testing: learning rate")
    plt.savefig('tests/figs/linear_learning_rate.png')
    plt.show()

    # Test different batch sizes
    batch_sizes = [1, 10, 100]

    plt.figure(figsize=(12,8))
    for i, batch_size in enumerate(batch_sizes):
        # Initialise hyperparams
        n_epochs = 200
        lr = 0.01
        seed = 13
        # Obtain data
        X_train, y_train = prepare_data(slope = 5, normalize = True, seed = seed)
        # Initialise model and loss 
        lin_module = Linear(1, 1, seed = seed)
        mse_loss = MSELoss()
        # Train
        all_losses, all_params = train_model(X_train, y_train, lin_module, mse_loss, lr, batch_size, n_epochs, seed = seed, verbose = False)
        
        # Visualization
        plt.subplot(len(batch_sizes), 3, i*3 + 1)
        plt.plot(all_losses)
        plt.ylabel(f"batch_size = {batch_size}\n\nLoss (MSE)")
        plt.xlabel("Epoch")
        plt.ylim(bottom = 0)
        if i == 0: plt.title("Loss over epochs")
        
        plt.subplot(len(batch_sizes), 3, i*3 + 2)
        plt.plot(all_params)
        plt.ylabel(f"Model Weight")
        plt.xlabel("Epoch")
        if i == 0: plt.title("Model Param")
        
        m_hat = all_params[-1]
        predictions = m_hat * X_train
        plt.subplot(len(batch_sizes), 3, i*3 + 3)
        plt.scatter(X_train, y_train)
        plt.plot(X_train, predictions, color = 'red', label = f'Prediction: y = {m_hat: .2f}')
        plt.legend()
        if i == 0: plt.title(f"Linear Regression against Training Data")

    plt.suptitle("Hyperparameter testing: batch size")
    plt.savefig('tests/figs/linear_batch_size.png')
    plt.show()
