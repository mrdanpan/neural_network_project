import numpy as np
from modules import Linear
from losses import MSELoss
import matplotlib.pyplot as plt

X_train = np.linspace(0, 10, 50).reshape(-1, 1)
y_train = 5 * X_train + np.random.normal(size=(X_train.shape), scale=5)
plt.scatter(X_train, y_train)
plt.show()

lin_module = Linear(1, 1)
mse_loss = MSELoss()

all_losses = []

n_epochs = 100
for epoch in range(n_epochs):
    lin_module.zero_grad()
    permutation = list(range(len(X_train)))
    np.random.shuffle(permutation)

    epoch_loss = 0

    for i in permutation:
        lin_module.zero_grad()
        X, y = X_train[i], y_train[i]

        # forward
        y_pred = lin_module.forward(X)
        loss = mse_loss.forward(y, y_pred)
        epoch_loss += loss

        # print(f'y: {y}; y_pred: {y_pred}, loss={loss}')

        # backward
        delta_loss = mse_loss.backward(y, y_pred)
        lin_module.backward_update_gradient(X, delta_loss)
        # no need to call lin_module.backward_delta()
        lin_module.update_parameters(0.00002)

    # update parameters
    # print("Epoch Loss:", epoch_loss)
    all_losses.append(epoch_loss[0])

    # lin_module.update_parameters(gradient_step=0.00005)
    # print("current params", lin_module._parameters)

plt.plot(list(range(n_epochs)), all_losses)
plt.show()