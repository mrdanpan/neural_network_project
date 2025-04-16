import path_config
from neural_network.modules import *
from neural_network.losses import *
from neural_network.optim import *
from neural_network.train_gen import MBGD
from neural_network.util import *
import numpy as np 
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


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

X_train, y_train = np.array([x.flatten() for x in X_train])[0:5000], np.array(y_train)[0:5000]
X_test, y_test = np.array([x.flatten() for x in X_test])[0:5000], np.array(y_test)[0:5000]

y_train_onehot = np.zeros(shape = (np.array(y_train).shape[0],10))
y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

y_test_onehot = np.zeros(shape = (np.array(y_test).shape[0],10))
y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1

# Model training
model = Sequential([Linear(28 * 28, 100), Sigmoid(), Linear(100, 10), SoftMax()])
loss = CrossEntropy()
optim = Optim(model, loss, eps = 1e-4)

all_losses = MBGD(X_train, y_train_onehot, model, loss, optim, batch_size = 100, nb_epochs = 100, seed = 10)

preds = model.forward(X_test[0:10])
print(preds)
print(y_test_onehot[0:10])


