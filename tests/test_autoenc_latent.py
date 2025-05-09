import pickle
import numpy as np
import path_config as path_config
from neural_network.modules import *
from neural_network.losses import *
from neural_network.optim import *
from neural_network.train_gen import *
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt

# Train/test split
def get_split_indices(X, perc_test=0.2, seed=10):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - perc_test))
    return idx[:split], idx[split:]
def load_weights(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare_weights(path1, path2, atol=1e-6):
    weights1 = load_weights(path1)
    weights2 = load_weights(path2)

    if len(weights1) != len(weights2):
        print(f"Length mismatch: {len(weights1)} vs {len(weights2)}")
        return False

    all_match = True
    for i, (w1, w2) in enumerate(zip(weights1, weights2)):
        if w1 is None and w2 is None:
            continue
        elif w1 is None or w2 is None:
            print(f"Mismatch at index {i}: one is None, the other is not.")
            all_match = False
        elif not np.allclose(w1, w2, atol=atol):
            print(f"Mismatch at index {i}: arrays differ.")
            all_match = False
        else:
            print(f"Index {i}: OK")

    return all_match


if __name__ == "__main__":
    ## % Data Preparation
    autoenc_dir = 'tests/autoencoder_results/'
    custom_transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(),
        transforms.ToTensor()           
    ])
    dataset = FashionMNIST(root='./data', download=True, transform=custom_transform)
    small_dataset = Subset(dataset, range(10000))

    X, _ = zip(*[small_dataset[i] for i in range(len(small_dataset))])
    X_np = np.array([x.numpy().flatten().astype(np.float32) for x in X])  

    train_idx, test_idx = get_split_indices(X, perc_test=0.2, seed=10)
    X_train_np, X_test_np = X_np[train_idx], X_np[test_idx]

    weights_path = 'tests/autoencoder_results/autoencoder_params_MSE_epoch2000.pkl'
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)

    path1 = 'tests/autoencoder_results/autoencoder_params_MSE_epoch0.pkl'
    path2 = 'tests/autoencoder_results/autoencoder_params_MSE_epoch2000.pkl'
    match = compare_weights(path1, path2)

    print("\nResult:", "Weights match!" if match else "Weights differ...)")


    num_pixels = np.matrix.flatten(X_train_np[0]).shape[0] # 786
    hidden1_num_neurons = 256
    hidden2_num_neurons = 64
    hidden3_num_neurons = 2

    autoencoder = Sequential([
                            # Encoder
                            Linear(num_pixels, hidden1_num_neurons, weight_initialisation="He"), TanH(), 
                            Linear(hidden1_num_neurons, hidden2_num_neurons, weight_initialisation="He"), TanH(),
                            Linear(hidden2_num_neurons, hidden3_num_neurons, weight_initialisation="He"), TanH(),
                            
                            # Decoder
                            Linear(hidden3_num_neurons, hidden2_num_neurons, weight_initialisation="He"), TanH(),
                            Linear(hidden2_num_neurons, hidden1_num_neurons, weight_initialisation="He"), TanH(), 
                            Linear(hidden1_num_neurons,num_pixels, weight_initialisation="He"), Sigmoid()
                            ])
    for i, layer in enumerate(autoencoder.modules):
        layer._parameters = weights[i]

    def add_noise(images, scaling_factor = 0.01):
        images_copy = []
        for i, im in enumerate(images):
            images_copy.append((im.copy() +  np.random.randn(im.shape[0]) * scaling_factor))
        return np.array(images_copy)

    # NO NOISE
    autoenc_dir = 'tests/autoencoder_results/'
    plt.figure()
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(transforms.ToPILImage()(X_test_np[i].reshape(28,28)))
        plt.xticks([]); plt.yticks([])
    plt.suptitle("No noise")
    plt.savefig(autoenc_dir + "Xtest_no_noise")
    plt.close()

    preds_nonoise = autoencoder.forward(X_test_np)
    plt.figure(figsize=(12,5))
    for i in range(10):
        plt.subplot(2,10, i+1)
        plt.imshow(transforms.ToPILImage()(X_test_np[i].reshape(28,28)))
        plt.xticks([]); plt.yticks([])
    for i in range(10,20):
        plt.subplot(2,10, i + 1)
        plt.imshow(preds_nonoise[i].reshape(28,28))
        plt.xticks([]); plt.yticks([])
    plt.suptitle(f'Noise level: 0')


    # NOISE
    losses = []
    mse = MSELoss()

    scaling_factors = [0, 0.05, 0.1, 0.5, 1, 1.5]
    for scaling_factor_i in scaling_factors:
        X_test_np_noised = add_noise(X_test_np, scaling_factor = scaling_factor_i)
        preds_noised = autoencoder.forward(X_test_np_noised)
        losses.append(mse.forward(X_test_np_noised, preds_noised))
        
        plt.figure(figsize=(12,3))
        for i in range(10):
            plt.subplot(2,10, i+1)
            plt.imshow(transforms.ToPILImage()(X_test_np_noised[i].reshape(28,28)))
            plt.xticks([]); plt.yticks([])
        for i in range(10,20):
            plt.subplot(2,10, i + 1)
            plt.imshow(preds_noised[i].reshape(28,28))
            plt.xticks([]); plt.yticks([])
        plt.suptitle(f'Noise level: {scaling_factor_i}')
        name_im = autoenc_dir + f"predictions_with_Xtest_noise_level_{scaling_factor_i}.png"
        plt.savefig(name_im)
            

    plt.figure(figsize=(6,6))
    plt.plot(scaling_factors, losses)
    plt.xticks(scaling_factors, rotation = 45)
    plt.ylabel("MSE Loss")
    plt.xlabel("Noise scaling factor")
    plt.title("Effect of noisy inputs on loss (all test examples)")
    name_im = autoenc_dir + f"losses_in_function_of_scaling_factor.png"
    plt.savefig(name_im)

    # def p_ji(pred_i, pred_j, predictions, sigma_i = 5):
    #     num = np.exp(- np.linalg.norm(pred_i - pred_j)**2 / (2 * sigma_i**2))
        
    #     denom = 0
    #     for pred_k in predictions:
    #         if not np.allclose(pred_k, pred_i):
    #             denom += np.exp(- np.linalg.norm(pred_i - pred_k)**2 / (2 * sigma_i**2))
                
    #     return num / denom

    # def q_ji(pred_i, pred_j, predictions):
    #     num = (1 + np.linalg.norm(pred_i - pred_j)**2)**(-1)
        
    #     denom = 0
    #     for pred_k in predictions:
    #         for pred_l in predictions:
    #             if not np.allclose(pred_k, pred_l):
    #                 denom += (1 + np.linalg.norm(pred_k - pred_l)**2)**(-1)
                
    #     return num / denom
            
    # def distribution_P(predictions):
    #     P = np.zeros(shape=(predictions.shape[0], predictions.shape[0]))
    #     for i, pred_i in enumerate(predictions):
    #         for j, pred_j in enumerate(predictions):
    #             P[i][j] =p_ji(pred_i, pred_j, predictions, sigma_i = 1)
    #     return P


    # def distribution_Q(Y):
    #     Q = np.zeros(shape=(Y.shape[0], Y.shape[0]))
    #     for i, pred_i in enumerate(Y):
    #         for j, pred_j in enumerate(Y):
    #             Q[i][j] =q_ji(pred_i, pred_j, Y)
    #     return Q

    # def KL_distribution(P, Q, eps=1e-10):
    #     mask = (P > 0)
    #     return np.sum(P[mask] * np.log((P[mask] + eps) / (Q[mask] + eps)))


    # dim_Y = 2
    # Y = np.random.randn(latent_preds.shape[0], dim_Y)

    # def tSNE_step(Y, P, learning_rate=200.0):
    #     N, dim = Y.shape
    #     Q = distribution_Q(Y)
    #     grads = np.zeros_like(Y)

    #     for i in range(N):
    #         grad_i = np.zeros(dim)
    #         for j in range(N):
    #             if i != j:
    #                 diff = Y[i] - Y[j]
    #                 q_ij = (1 + np.linalg.norm(Y[i] - Y[j])**2)**-1
    #                 grad_i += 4 * (P[i, j] - Q[i, j]) * q_ij * diff
    #         grads[i] = grad_i

    #     # Update Y
    #     Y -= learning_rate * grads
    #     return Y, Q


    # latent_preds = encoder.forward(X_test_np[:100])  # smaller subset for speed
    # P = distribution_P(latent_preds)

    # dim_Y = 2
    # Y = np.random.randn(latent_preds.shape[0], dim_Y) * 1e-4

    # for step in tqdm(range(500)):  # number of gradient descent steps
    #     Y, Q = tSNE_step(Y, P, learning_rate=10.0)
    #     if step % 50 == 0:
    #         kl = KL_distribution(P, Q)
    #         print(f"Step {step}: KL divergence = {kl:.4f}")


    # plt.scatter(Y[:, 0], Y[:, 1])
    # plt.title("2D t-SNE projection")
    # plt.show()

