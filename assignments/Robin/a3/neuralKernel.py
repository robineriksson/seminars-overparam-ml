##' Robin Eriksson 2021-11-16
##'
##' Dependencies:
##'    scipy      (1.7.1)
##'    numpy      (1.19.5)
##'    matplotlib (3.4.3)
##'    torch  (1.10.0+cu102)
##'    tqdm  (4.62.3)


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.stats import multivariate_normal
import scipy.linalg as linalg
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm




# Small value it will be added to the covariance matrix to avoid numerical
# instabilities yielding a non positive definite covariacne matrix
EPS = 1e-7

## -------------------------------------------------------------------------------
## Taken from: nerual_network.py
## -------------------------------------------------------------------------------
class FeedforwardNetwork(nn.Module):
    """Neural network implemented using NumPy."""

    def __init__(self, input_size, width=1000, beta=0.1, depth=4):
        super(FeedforwardNetwork, self).__init__()
        self.width = width
        self.beta = beta
        self.depth = depth
        self.ws = [input_size] + [self.width for _l in range(self.depth)] + [1]

        self.weights = []
        self.biases = []
        for i in range(depth + 1):
            wi = nn.Parameter(torch.randn(self.ws[i], self.ws[i + 1]))
            bi = nn.Parameter(torch.randn(self.ws[i + 1]))
            self.register_parameter('weight_{}'.format(i), wi)
            self.register_parameter('bias_{}'.format(i), bi)
            self.weights += [wi]
            self.biases += [bi]

    def forward(self, X):
        """Return output"""
        Xf = self.get_features(X)
        pred = 1/np.sqrt(self.ws[-2]) * Xf @ self.weights[-1] + self.beta * self.biases[-1]
        return pred.flatten()

    def get_features(self, X):
        activ = X
        for i in range(self.depth):
            preactiv = 1/np.sqrt(self.ws[i]) * activ @ self.weights[i] + self.beta * self.biases[i]
            activ = torch.relu(preactiv)
        return activ
## -------------------------------------------------------------------------------
## Taken from: gaussian_process.py
## -------------------------------------------------------------------------------
def conditioning(c, y_obs):
    n_obs = len(y_obs)
    if n_obs == 0:
        return np.zeros(c.shape[0]), c
    cov_obs = c[:n_obs, :n_obs]
    cov_eval_obs = c[n_obs:, :n_obs]
    cov_eval = c[n_obs:, n_obs:]
    inv_cov = np.linalg.inv(cov_obs)
    m = cov_eval_obs @ inv_cov @ y_obs
    cc = cov_eval - cov_eval_obs @ inv_cov @ cov_eval_obs.T
    return m, cc

## -------------------------------------------------------------------------------
## Part 1
## -------------------------------------------------------------------------------

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
    K = K + 5*EPS*np.eye(K.shape[0]) ## first one seems a bit hard
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

def part1(samples=5,savefig=True):
    '''
    Part 1: Neural Network as a Gaussian Process

    Given observation from the function f = cos(x)*sin(x)
    we train a regression model to approximate the function.

    Generates 5 samples from
    1) Only final layer trained network (lsqfit)
    2) Neural Network approximated kernel Gaussian Process

    For the latter Credible intervals (68,95,99) are included in the plot.

    Inputs:
    samples -- number of samples from NN and GP in the plot.

    Outputs:
    generates a pyplot, nothing returned.
    '''
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
    for i in range(samples):
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
    m, cc = conditioning(c, Y_obs) # see gaussian_process

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
    plt.legend(loc='upper right')
    if savefig:
        plt.savefig("part1.png",bbox_inches='tight')
    else:
        plt.show()

## -------------------------------------------------------------------------------
## Part 2
## -------------------------------------------------------------------------------

def gradient_train(net, epochs, lr, X_obs, Y_obs):
    '''
    Given the network, compute the l2 gradient descent and update the
    weights

    input:
    net -- pytorch network
    epochs -- number of data passthroughs
    lr -- learning rate
    X_obs -- input observations
    Y_obs -- output observations
    returns:
    net -- trained network
    '''
    msg = 'Epoch = {} - loss = {:0.2f}'
    pbar = tqdm.tqdm(initial=0, total=epochs, desc=msg.format(0, 0))
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for i in range(epochs):
        optimizer.zero_grad()
        Y_pred = net(X_obs)
        error = (Y_obs - Y_pred)
        loss = 1/2 * (error * error).mean()
        loss.backward()
        optimizer.step()
        # Update pbar
        pbar.desc = msg.format(i, loss)
        pbar.update(1)
    pbar.close()

    return net

def reludot(x):
    '''
    returns the derivative of the relu
    '''
    return (x>0).astype(x.dtype)

def NTK(X,beta=0.1,n_layers=1,size=1000, EPS_scale=1):
    '''
    Approximates the Neural tangent kernel

    input:
    X -- All available data
    beta -- kernel parameters
    n_layers -- number of layers in network
    size -- number of samples used to estimate the covariance/kernel
    EPS_scale -- a scaling factor for the global EPS used to get
                 around non semi-definite matrices

    returns:
    theta -- Neural tangent kernel

    '''
    n0 = X.shape[1]
    K = 1/n0*(X @ X.T) + beta**2
    ##K = K + EPS*np.eye(K.shape[0]) ## first one seems a bit hard

    theta = K

    def fillkernel(cov,beta,K):
        K_ = np.zeros(K.shape)
        for i in range(cov.shape[1]):
            for j in range(cov.shape[1]):
                 K_[i,j]= np.mean(cov[:,i]*cov[:,j]) + beta**2
        return K_

    for l in range(n_layers-1): ## per layer
        Z = multivariate_normal(mean=None, cov=K+ EPS_scale*EPS*np.eye(K.shape[0])).rvs(size)
        sig = relu(Z)
        sigdot = reludot(Z)


        ## fill the kernel
        K = fillkernel(sig,beta,K)
        Kdot = fillkernel(sigdot,beta,K)

        ## update theta
        theta = theta*Kdot + K

    return theta

def NTK_train(X_obs,X_eval,Y_obs,beta,n_layers,size,samples):
    '''
    Estimates the NTK trained network.


    '''


    X_all = np.vstack([X_obs, X_eval])
    theta = NTK(X=X_all,beta=beta,n_layers=n_layers,size=1000,EPS_scale=5)
    sig = kernel_approx(X_all,beta=beta,n_layers=n_layers,size=size)

    n_obs = X_obs.shape[0]
    ## unwrap theta
    cov_obs = theta[:n_obs, :n_obs]
    cov_eval_obsT = theta[n_obs:, :n_obs]
    cov_eval = theta[n_obs:, n_obs:]

    ## same goes for sigma
    sig_obs = sig[:n_obs, :n_obs]
    sig_eval_obsT = sig[n_obs:, :n_obs]
    sig_eval = sig[n_obs:, n_obs:]

    ## The Neural tangent kernel trained network is basically as Normal distribution
    cov_obs_inv = linalg.inv(cov_obs)

    Y_obs = Y_obs.numpy()

    mu = cov_eval_obsT @ cov_obs_inv @ Y_obs


    C = sig_eval  +\
        cov_eval_obsT @ cov_obs_inv @ sig_obs @ cov_obs_inv @ cov_eval_obsT.T -\
        cov_eval_obsT @ cov_obs_inv @ sig_eval_obsT.T -\
        sig_eval_obsT @ cov_obs_inv @ cov_eval_obsT.T

    feval = multivariate_normal(mean=mu, cov=C).rvs(samples)

    return feval, mu, C



def part2(samples=5,savefig=True):
    '''
    Part 2: Neural Tangent Kernel

    Given observation from the function f = cos(x)*sin(x)
    we train a regression model to approximate the function.

    Generates 5 samples from
    1) Gradient descent trained (wide) neural network
    2) Neural tangent trained neural network (we can skip to final step as the training is
       deterministic).

    For the latter Credible intervals (68,95,99) are included in the plot.

    Inputs:
    samples -- number of samples from the two NN

    Outputs:
    generates a pyplot, nothing returned.
    '''
    N_eval = 100
    n_layers = 4
    beta = 0.1
    epochs = 1000
    width = 1000
    lr = 1 # learning rate
    ## seed = 1  # Change here for neural networks with different initializations
    size = 1000

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

    # Define neural network
    for seed in range(samples):
        torch.manual_seed(seed)
        net = FeedforwardNetwork(2, beta=beta, depth=n_layers, width=width)



        net = gradient_train(net, epochs=epochs, lr=lr,
                             X_obs=X_obs_pth, Y_obs=Y_obs_pth)



        # Predict on test
        with torch.no_grad():
            Y_pred_eval_grad = net(X_eval_pth).detach().numpy()

        # Plot on test data
        if seed > 0:
            plt.plot(gamma_eval, Y_pred_eval_grad, color='red',linestyle="dashed")
        else:
            plt.plot(gamma_eval, Y_pred_eval_grad, color='red',linestyle="dashed",label='w/ GD')

    # NTK training
    Y_pred_eval,m,cc  = NTK_train(X_obs=X_obs_pth,X_eval=X_eval_pth,Y_obs=Y_obs_pth,
                            beta=beta,n_layers=n_layers,size=size,samples=samples)
    plt.plot(gamma_eval, Y_pred_eval[0,:], color='grey',label='w/ NTK')
    plt.plot(gamma_eval, Y_pred_eval[1:,:].T, color='grey')

    # credible intervals
    for nn in [3, 2, 1]:
        plt.fill_between(gamma_eval, m - nn*np.sqrt(np.diag(cc)),
                         m + nn*np.sqrt(np.diag(cc)), color=str(0.4+0.15 * nn), alpha=0.5)

    # rest of data
    plt.plot(gamma_eval, Y_eval, color='blue',label='TRUE')
    plt.plot(gamma_obs, Y_obs, '*', color='black', ms=10,label='Observed')
    plt.xlabel('gamma')
    plt.ylabel('f')
    plt.legend()
    if savefig:
        plt.savefig("part2.png",bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    part1(5,True)
    part2(5,True)
