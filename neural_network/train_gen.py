import numpy as np
from tqdm import tqdm

def MBGD(X_train, y_train, model, loss_class, optimizer, batch_size = 10, nb_epochs = 100, seed = None, verbose = True, save_params = False):
    all_losses = []
    if save_params: all_params = [model._parameters.copy()]
    
    for epoch in tqdm(range(nb_epochs)):
        model.zero_grad()
        
        if seed is not None:
            np.random.seed(seed + epoch)
            
        permutation = list(range(len(X_train)))
        np.random.shuffle(permutation)

        epoch_loss = 0
        
        # In MB, last batch will be smaller than batch_size if len(X_train) % batch_size != 0 
        for i in range(0, len(permutation), batch_size):
            model.zero_grad()
            X, y = X_train[permutation[i: i + batch_size]], y_train[permutation[i: i + batch_size]]

            # forward
            y_pred = model.forward(X)

            loss = loss_class.forward(y, y_pred)
            epoch_loss += np.mean(loss)

            # backward
            optimizer.step(X, y)
            
        # update training metrics 
        all_losses.append(np.mean(epoch_loss))
        if save_params: all_params.append(model._parameters.copy())
        
        if verbose: 
            print(f"Epoch {epoch} Loss: {epoch_loss}")
    if save_params: return all_losses, all_params
    else: return all_losses
    

# Note: this is the same thing as doing MBGD with batch_size = 1. 
def SGD(X_train, y_train, model, loss_class, optimizer, nb_epochs = 100, batch_size = 1, seed = None, verbose = True):
    
    all_losses = []
    
    for epoch in range(nb_epochs):
        model.zero_grad()
        epoch_loss = 0
        
        # In SGD, we pick one example out of the training set 
        if seed is not None:
            np.random.seed(seed + epoch)
        idx = np.random.randint(0, len(X_train))
        X, y = X_train[idx], y_train[idx]

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
