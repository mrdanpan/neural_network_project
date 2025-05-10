import numpy as np
from tqdm import tqdm
import copy

def SGD(X_train, y_train, model, loss_class, optimizer, batch_size = 10, nb_epochs = 100, seed = None, verbose = True, save_params = False):
    all_losses = []
    if save_params: all_params = [copy.deepcopy(model._parameters)]
    
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
        if save_params: all_params.append(copy.deepcopy(model._parameters))
        
        if verbose: 
            print(f"Epoch {epoch} Loss: {epoch_loss}")
    if save_params: return all_losses, all_params
    return all_losses
