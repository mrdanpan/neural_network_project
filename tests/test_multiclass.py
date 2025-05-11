import path_config
from neural_network.modules import *
from neural_network.losses import *
from neural_network.optim import *
from neural_network.train_gen import SGD
from neural_network.util import *
import numpy as np 
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# Data preparation
seed = 10
np.random.seed(seed = seed)
num_examples, num_classes = 5, 4  # 5 examples, 4 classes
X = np.random.random(size = (num_examples, num_classes))
y = np.zeros(shape=(num_examples, num_classes)) # onehot vector: 5 examples
idxs_ones = np.random.randint(0, num_classes, size = (num_examples))
for i, row in enumerate(y):
    row[idxs_ones[i]] = 1 

# Test the effects of softmax, with and without log 
sm_X_module = SoftMax().forward(X)
sm_X_torch = torch.nn.Softmax(dim = 1).forward(torch.from_numpy(X))
assert np.allclose(sm_X_module, sm_X_torch.numpy()), "Softmax Module and Torch Softmax Module don't give the same result!"

sm_X_module_log = SoftMax(log=True).forward(X)
sm_X_torch_log = torch.nn.LogSoftmax(dim = 1).forward(torch.from_numpy(X))
assert np.allclose(sm_X_module_log, sm_X_torch_log.numpy()), "Softmax Module and Torch Softmax Module don't give the same result!"

# Testing CE, with and without log
logits = np.round(X, 4) # since torch-ifying numpys rounds to 4th decimal place
sm_X_module = SoftMax().forward(logits)
loss_module = CrossEntropy().forward(y, sm_X_module, log = False) # Our module uses the softmax as input
loss_torch = torch.nn.CrossEntropyLoss(reduction = 'none').forward(torch.from_numpy(logits),torch.from_numpy(y)) # torch uses logits as input and calculates it internally

assert np.allclose(logits, torch.from_numpy(logits))
assert np.allclose(y, torch.from_numpy(y))
assert np.allclose(loss_module, loss_torch.numpy())

y_indexs = np.where(y == 1)[1] # For NLL, which doesn't take one hot...
sm_X_module_log = np.round(SoftMax(log=True).forward(logits),4)
loss_module_log = CrossEntropy().forward(y, sm_X_module_log, log = True) 
loss_torch_log = torch.nn.NLLLoss(reduction = 'none').forward(torch.from_numpy(sm_X_module_log), torch.from_numpy(y_indexs))
assert np.allclose(loss_module_log, loss_torch_log)

# Testing backward functions: pytorch does it in term of logits, so gradients are not directly comparable...
## Testing on MNIST
# Data prep
X_train, y_train = zip(*MNIST(root='./data', train=True, download=True, transform=ToTensor()))
X_test, y_test = zip(*MNIST(root='./data', train=False, download=True, transform=ToTensor()))

size_ds = 10000 # modify in terms of computational load
X_train, y_train = np.array([x.flatten() for x in X_train])[0:size_ds], np.array(y_train)[0:size_ds]
X_test, y_test = np.array([x.flatten() for x in X_test])[0:size_ds], np.array(y_test)[0:size_ds]

y_train_onehot = np.zeros(shape = (np.array(y_train).shape[0],10))
y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

y_test_onehot = np.zeros(shape = (np.array(y_test).shape[0],10))
y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1


# Function for quality check
def show_model_pred(X_test, y_test_onehot, nb_epochs, save_path = None):
    range_min = 100; range_max = 105
    preds = model.forward(X_test[range_min:range_max])
    plt.figure(figsize = (4,8))
    for i, (image, pred) in enumerate(zip(X_test[range_min:range_max], preds)):
        true_label = np.argmax(y_test_onehot[range_min + i])
        
        
        plt.subplot(5, 2, i*2 + 1)
        plt.imshow(image.reshape(28,28), cmap = 'grey')
        plt.xticks([]); plt.yticks([])
        
        plt.subplot(5, 2, i*2 + 2)
        plt.bar(np.arange(10), pred)    
        plt.xticks(np.arange(10))
        plt.xlim(left = -1) 
        plt.yticks(np.arange(0,1.25,0.25))
        plt.grid(True)
        
        ax = plt.gca()
        xticks = ax.get_xticklabels()
        xticks[true_label].set_color('red')
        
    plt.suptitle(f"Model predictions on test set after {nb_epochs} epochs")
    if save_path: 
        fig_im = f'{save_path}multiclass_predictions_{nb_epochs}epochs.png'
        plt.savefig(fig_im)
    plt.show()
    
# Model training
model = Sequential([Linear(28 * 28, 100, weight_initialisation='He'), Sigmoid(), Linear(100, 10, weight_initialisation='He'), SoftMax()])
loss = CrossEntropy()
optim = Optim(model, loss, eps = 1e-4)

show_model_pred(X_test, y_test_onehot, nb_epochs=0, save_path='tests/figs/')
all_losses = SGD(X_train, y_train_onehot, model, loss, optim, batch_size = 100, nb_epochs = 500, seed = 10)
show_model_pred(X_test, y_test_onehot, nb_epochs=500, save_path='tests/figs/')


plt.figure(figsize=(12,6))
plt.plot(all_losses)
plt.title('CE Loss over Epochs')
plt.xlabel("Epoch")
plt.ylabel("CE Loss")
plt.savefig('tests/figs/multiclass_model_loss.png')