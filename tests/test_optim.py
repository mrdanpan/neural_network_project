import path_config
from neural_network.losses import *
from neural_network.modules import * 
from neural_network.optim import * 
import matplotlib.pyplot as plt 
from test_linear import train_model as train_model_without_optim


def train_model(X_train, y_train, model, loss_class, optimizer, batch_size = 1, nb_epochs = 100, seed = None, verbose = True):
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
            optimizer.step(X, y)
        # update training metrics 
        all_losses.append(np.mean(epoch_loss))
        
        if verbose: 
            print(f"Epoch {epoch} Loss: {epoch_loss}")

    return all_losses

# Hyperparams 
nb_epochs = 200
batch_size = 1
seed = 13
lr = 0.01

net = Linear(1, 1, seed = seed)
loss = MSELoss()

# Data set up
def prepare_data(slope, normalize = True, seed = 10):
        np.random.seed(seed)
        X_train = np.linspace(0, 10, 50).reshape(-1, 1)
        y_train = slope * X_train + np.random.normal(size=(X_train.shape), scale=5)

        if normalize: 
            X_train = (X_train - np.mean(X_train)) / np.std(X_train)
            y_train = (y_train - np.mean(y_train)) / np.std(y_train)
        return X_train, y_train
    
X, y = prepare_data(slope = 5, normalize = True, seed = seed)

all_losses, _  = train_model_without_optim(
            X_train = X, y_train = y, 
            model = net, loss_class = loss, learning_rate = lr,
            batch_size = batch_size, n_epochs = nb_epochs, seed = seed, 
            verbose = False
            )


net2 = Linear(1, 1, seed = seed) # Reinitialize model params 
loss2 = MSELoss()
optim = Optim(net2, loss, eps = lr)
all_losses_optim = train_model(
            X_train = X, y_train = y, 
            model = net2, loss_class = loss2, optimizer = optim, 
            batch_size = batch_size, nb_epochs = nb_epochs, seed = seed, 
            verbose = False
            )
            
# Without optim
plt.subplot(2,2,1)
plt.plot(all_losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Loss over epochs")

m_hat = net._parameters[0][0]
predictions = m_hat * X
plt.subplot(2,2,2)
plt.scatter(X, y)
plt.plot(X, predictions, color = 'red', label = f'Prediction: y = {m_hat: .2f}')
plt.xlim([-2,2]); plt.ylim([-2,2])
plt.legend()
plt.title(f"Linear Regression against Training Data")

# With optim
plt.subplot(2,2,3)
plt.plot(all_losses_optim)
plt.ylabel("Loss")
plt.xlabel("Epoch")

m_hat = net2._parameters[0][0]
predictions = m_hat * X
plt.subplot(2,2,4)
plt.scatter(X, y)
plt.plot(X, predictions, color = 'red', label = f'Prediction: y = {m_hat: .2f}')
plt.legend()
plt.xlim([-2,2]); plt.ylim([-2,2])

plt.show()
        
        

