##' Robin Eriksson 2021-11-16
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.stats import multivariate_normal

from neural_network import FeedforwardNetwork
from gaussian_process import conditioning
import torch

# Small value it will be added to the covariance matrix to avoid numerical
# instabilities yielding a non positive definite covariacne matrix
EPS = 1e-7


## Part 1.1
def finalLayer_fit(net,X,Y,beta=0.1):
    '''
    Fit the final layer using least squares regression

    note the scaling needed for the weights and the bias.
    '''
    nl = net.weights[-1].shape[0]

    ## get the activation in the last layer
    fths = net.get_features(X).detach().numpy()
    fths = fths / np.sqrt(nl)
    A0 = np.c_[fths,beta*np.ones(fths.shape[0])]

    ## fit parameters
    param,_,_,_ = lstsq(A0,Y)

    ## adjust final layer weights and bias
    net.weights[-1] = param[:-1]
    net.biases[-1] = param[-1]

    return net

## Part 1.2
def relu(x):
    '''
    Computes the classical ReLU on input x, works on numpy matrices per element
    '''
    return np.maximum(x,0)

def kernel_approx(X,beta=0.1,n_layers=1,size=1000):
    '''
    Neural network approximated Kernel.
    Given data X, and Network parameters, approximate the Kernel
    "recursively".
    '''
    n0 = X.shape[1]
    K = 1/n0*(X @ X.T) + beta**2
    K = K + EPS*np.eye(K.shape[0]) ## first one seems a bit hard
    for l in range(n_layers-1): ## per layer
        Z = multivariate_normal(mean=None, cov=K).rvs(size)
        sig = relu(Z)

        ## fill the kernel
        K_ = np.zeros(K.shape)
        for i in range(sig.shape[1]):
            for j in range(sig.shape[1]):
                 K_[i,j]= np.mean(sig[:,i]*sig[:,j]) + beta**2
        K = K_

    return K

def main(seeding=5):
    N_eval = 100
    n_layers = 4
    beta = 0.1
    epochs = 1000
    width = 1000
    lr = 1
    K = 1000 # GP samples

    # Train data
    gamma_obs = np.array([-2, -1.2, -0.4, 0.9, 1.8])
    X_obs = np.stack([np.cos(gamma_obs), np.sin(gamma_obs)]).T
    Y_obs = X_obs.prod(axis=1)

    # Test data
    gamma_eval = np.linspace(-np.pi, np.pi, N_eval)
    X_eval = np.stack([np.cos(gamma_eval), np.sin(gamma_eval)]).T
    Y_eval = X_eval.prod(axis=1)

    # Convert tensors from numpy to pytorch
    X_obs_pth = torch.Tensor(X_obs)
    Y_obs_pth = torch.Tensor(Y_obs)
    X_eval_pth = torch.Tensor(X_eval)

    ## Train only the final layer
    for i in range(seeding):
        # Define neural network
        torch.manual_seed(i) ## for NN
        net = FeedforwardNetwork(2, beta=beta, depth=n_layers, width=width)

        # Predict on test | with training
        net = finalLayer_fit(net=net,X=X_obs_pth,Y=Y_obs_pth,beta=beta)
        with torch.no_grad():
            Y_pred_eval = net(X_eval_pth).detach().numpy()
        # Plot on test data
        if i == 0:
            plt.plot(gamma_eval, Y_pred_eval, color='red',linestyle='dashed',label="Final layer")
        else:
            plt.plot(gamma_eval, Y_pred_eval, color='red',linestyle='dashed')

    ## Approximate a GP using the wide NN
    np.random.seed(0) ## for GP
    X_all = np.vstack([X_obs, X_eval])
    c = kernel_approx(X_all,beta=beta,n_layers=n_layers,size=K)
    m, cc = conditioning(c, Y_obs)

    ## Sample the GP
    normal = multivariate_normal(mean=m.flatten(), cov=cc + EPS*np.eye(cc.shape[0]))
    Y_sols = normal.rvs(5)

    ## Finish plotting the results
    plt.plot(gamma_eval, Y_sols[0,:].T, color=str(0.4), ms=10, label='Neural network GP')
    plt.plot(gamma_eval, Y_sols[1:,:].T, color=str(0.4), ms=10)
    plt.plot(gamma_obs, Y_obs.T, '*', color='black', ms=10)
    plt.plot(gamma_eval, Y_eval.T, '-', color='black', ms=10)
    for nn in [3, 2, 1]:
        plt.fill_between(gamma_eval, m - nn*np.sqrt(np.diag(cc)),
                         m + nn*np.sqrt(np.diag(cc)), color=str(0.4+0.15 * nn), alpha=0.5)

    plt.plot(gamma_eval, Y_eval, color='blue',label='TRUE')
    plt.plot(gamma_obs, Y_obs, '*', color='black', ms=10,label='Observed')
    plt.xlabel('gamma')
    plt.ylabel('f')
    plt.legend()
    plt.show()







if __name__ == "__main__":
    main()
