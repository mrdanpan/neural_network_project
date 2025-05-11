import path_config
import numpy as np
import matplotlib.pyplot as plt
from neural_network.modules import * 
from neural_network.losses import *

from sklearn.metrics import roc_curve, RocCurveDisplay

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
    
    X_c1 = (np.array([c1 for _ in range(n)]).T + rng.normal(size = (len(c1), n))).T
    X_c2 = (np.array([c2 for _ in range(n)]).T + rng.normal(size = (len(c2), n))).T
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
    
    cutoff_idx = int(len(permutation)*(1 - perc_test))
    X_train, y_train = X[permutation[:cutoff_idx]], y[permutation[:cutoff_idx]]
    X_test, y_test = X[permutation[cutoff_idx:]], y[permutation[cutoff_idx:]]
    
    return X_train, y_train, X_test, y_test

def plot_data(TP, FP, TN, FN):

    if len(TP) > 0:
        plt.scatter(TP[:,0], TP[:,1], c='g', label='TP')
    if len(FP) > 0:
        plt.scatter(FP[:,0], FP[:,1], c='g', marker='x', label='FP')
    if len(TN) > 0:
        plt.scatter(TN[:,0], TN[:,1], c='r', label='TN')
    if len(FN) > 0:
        plt.scatter(FN[:,0], FN[:,1], c='r', marker='x', label='FN')
    
    plt.legend()
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title("Data points")
    
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
        all_losses.append(epoch_loss)

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

def model_score(model, X_in):
    out = X_in
    for layer in model:
        out = layer.forward(out)
    return out

def confusion_matrix(X_test, y_test, model, threshold=0.5, proportions = True, return_data = False):
    #[[TN, FP], [FN, TP]]
    y_pred = model_score(model, X_test)
        
    labels = np.unique(y_test)

    y_pred_bin = (y_pred >= threshold).astype(int).flatten()

    conf_mat = np.zeros((2, 2), dtype=int)  # [[TN, FP], [FN, TP]]

    tp, fp, tn, fn = [], [], [], []

    for i, (y_hat, y_true) in enumerate(zip(y_pred_bin, y_test)):
        if y_true == labels[1]:
            if y_hat == labels[1]:
                conf_mat[1, 1] += 1  # TP
                tp.append(X_test[i])
            else:
                conf_mat[1, 0] += 1  # FN
                fn.append(X_test[i])
        else:
            if y_hat == labels[1]:
                conf_mat[0, 1] += 1  # FP
                fp.append(X_test[i])
            else:
                conf_mat[0, 0] += 1  # TN
                tn.append(X_test[i])

    if proportions:
        conf_mat = conf_mat.astype(float)/y_test.shape[0]

    if not return_data:
        return conf_mat
    return conf_mat, (tp, fp, tn, fn)

def plot_confusion_matrix(conf_mat, title="Confusion Matrix", proportions=True):
    plt.imshow(conf_mat, cmap='viridis')
    plt.title(title)
    plt.colorbar()

    plt.xticks([0, 1], ['N', 'P'])
    plt.yticks([0, 1], ['N', 'P'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    for (i, j), val in np.ndenumerate(conf_mat):
        fmt = ".2f" if proportions else "d"
        plt.text(j, i, format(val, fmt), ha='center', va='center', color='black')


if __name__ == "__main__":
    # Hyperparams 
    seed = 10
    lr = 0.001
    batch_size = 1

    n_epochs = 10
    # Data preparation
    c1 = [1,2]; c2 = [3,5]
    X, y = prepare_binary_class_data(c2, c1, v1 = 0, v2 = 1, n = 100, normalize=True, seed = seed)
    X_train, y_train, X_test, y_test = split_train_test_data(X, y, perc_test=.2, seed = seed)

    all_losses_1, model_1 = train_model(X_train, y_train, nb_epochs=n_epochs, batch_size=1, lr = lr, seed = seed)
    print(f'Score for model with {n_epochs} training epochs: {score(X_test, y_test, model_1)} ')

    conf_mat_train, train_tup = confusion_matrix(X_train, y_train, model_1, proportions=True, return_data=True)
    conf_mat_test, test_tup = confusion_matrix(X_test, y_test, model_1, proportions=True, return_data=True)

    [TP, FP, TN, FN] = [np.array(train_tup[i]+test_tup[i]) for i in range(4)]
    
    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plot_data(TP, FP, TN, FN)
    plt.subplot(2,2,2)
    plot_loss(all_losses_1,  title = f"Losses for model trained on {n_epochs} epochs")
    plt.subplot(2,2,3)
    plot_confusion_matrix(conf_mat_train, title="Confusion Matrix (Train)", proportions=True)
    plt.subplot(2,2,4)
    plot_confusion_matrix(conf_mat_test, title="Confusion Matrix (Test)", proportions=True)
    plt.suptitle(f'Training on {n_epochs} epochs')
    plt.savefig(f'tests/figs/nonlin_{n_epochs}_epochs.png')
    plt.show()

    n_epochs = 200
    all_losses_2, model_2= train_model(X_train, y_train, nb_epochs=n_epochs, batch_size=1, lr = lr, seed = seed)
    print(f'Score for model with {n_epochs} training epochs: {score(X_test, y_test, model_2)} ')
    
    conf_mat_train, train_tup = confusion_matrix(X_train, y_train, model_2, proportions=True, return_data=True)
    conf_mat_test, test_tup = confusion_matrix(X_test, y_test, model_2, proportions=True, return_data=True)

    [TP, FP, TN, FN] = [np.array(train_tup[i]+test_tup[i]) for i in range(4)]

    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plot_data(TP, FP, TN, FN)
    plt.subplot(2,2,2)
    plot_loss(all_losses_2,  title = f"Losses for model trained on {n_epochs} epochs")
    plt.subplot(2,2,3)
    plot_confusion_matrix(conf_mat_train, title="Confusion Matrix (Train)", proportions=True)
    plt.subplot(2,2,4)
    plot_confusion_matrix(conf_mat_test, title="Confusion Matrix (Test)", proportions=True)
    plt.suptitle(f'Training on {n_epochs} epochs')
    plt.savefig(f'tests/figs/nonlin_{n_epochs}_epochs.png')
    plt.show()

    # plotting roc curve for test and train
    figure, axis = plt.subplots(2, 2, figsize=(10, 10))
    
    i = 0
    for m, model in enumerate([model_1, model_2]):
        y_train_hat = model_score(model, X_train)
        RocCurveDisplay.from_predictions(y_train, y_train_hat, ax=axis[i, 0])
        axis[i, 0].set_title(f'Model {m+1} Train ROC curve')

        y_test_hat = model_score(model, X_test)
        RocCurveDisplay.from_predictions(y_test, y_test, ax=axis[i, 1])
        axis[i, 1].set_title(f'Model {m+1} Test ROC curve')
        i += 1

    plt.show()