import path_config
from neural_network.losses import *
from neural_network.modules import * 
from neural_network.optim import * 
from neural_network.train_gen import * 
import matplotlib.pyplot as plt 

# Test linear to see if things are the same:
from test_linear import (
    prepare_data
)

if __name__ == "__main__":
        
    # Initialise data
    def prepare_data(slope, normalize = True, seed = 10):
        np.random.seed(seed)
        X_train = np.linspace(0, 10, 50).reshape(-1, 1)
        y_train = slope * X_train + np.random.normal(size=(X_train.shape), scale=5)

        if normalize: 
            X_train = (X_train - np.mean(X_train)) / np.std(X_train)
            y_train = (y_train - np.mean(y_train)) / np.std(y_train)
        return X_train, y_train

    # Test different learning rates
    lrs = [float(f'10e-{i}') for i in range(3,7)]

    plt.figure(figsize=(12,8))
    for i,lr in enumerate(lrs):
        # Initialise hyperparams
        nb_epochs = 200
        batch_size = 1
        seed = 13
        # Obtain data
        X_train, y_train = prepare_data(slope = 5, normalize = True, seed = seed)
        # Initialise model and loss 
        lin_module = Linear(1, 1, seed = seed)
        mse_loss = MSELoss()
        optimizer = Optim(lin_module, mse_loss, lr)
        # Train
        all_losses, all_params = SGD(X_train, y_train, lin_module, mse_loss, optimizer, nb_epochs = nb_epochs, seed = seed, save_params=True, verbose = False)
        all_params = [np.array(p).squeeze() for p in all_params]
        print(all_params[0:5])
        
        
        # Visualization
        plt.subplot(len(lrs), 3, i*3 + 1)
        plt.plot(all_losses)
        plt.ylabel(f"lr = {lr}\n\nLoss")
        plt.ylim(bottom = 0)
        plt.xlabel("Epoch")
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
        optimizer = Optim(lin_module, mse_loss, lr)
        # Train
        all_losses, all_params = SGD(X_train, y_train, lin_module, mse_loss, optimizer, nb_epochs = nb_epochs, seed = seed, save_params=True, verbose = False)
        all_params = [np.array(p).squeeze() for p in all_params]
        
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
    plt.show()
