##' Robin Eriksson 30/9-21

import numpy as np
from scipy.linalg import lstsq, norm
import matplotlib.pyplot as plt

def gen_data(N=300,p=50,r2=1,sigma2=1):
    """ 
    gendata generates data to use in the experiemnt 

    1) random inputs with isotropic inputs, N(loc,scale)
    2) Parameter vector with fixed l2 norm
    3) linear model
    4) train test spliting

    parameters
        N : int, number of samples in the input vector
        p : int, number of parameters
        r : float, norm restriction
        sigma2 : float, variance of noise in linear model
    returns
        X_train : numpy float, 2*N//3xP training input
        X_test : numpy float 1*N//3xP testing input
        y_train : numpy float, 2*N//3 training response
        y_test : numpy float, 1*N//3 testiong response
    """

    # 1) inputs
    x = np.random.normal(size=(N,p),loc=0,scale=1)

    # 2) parameter vector
    beta = np.random.normal(size=(p,1),loc=0,scale=np.sqrt(r2/p))

    # 3) model
    noise = np.random.normal(size=(N,1),loc=0,scale=np.sqrt(sigma2))
    y = (x @ beta) + noise

    # 4) train/test split
    X_train = x[:2*N//3,:]
    X_test = x[2*N//3:,:]
    y_train = y[:2*N//3]
    y_test = y[2*N//3:]
    
    return(X_train, X_test, y_train, y_test)

def fit_model(data=None):
    """
    Estimate paramters by minimum norm solution

    returns
    ------
    param : numpy float, p dim parameter array.
    """
    # unpack data
    if data is None:
        data = gendata()
    X_train, X_test, y_train, y_test = data

    # train model
    param,_,_,_ = lstsq(X_train,y_train)

    return param

def empirical_eval(param,data):
    """ 
    evaluate model fit
    """
    _, X_test, _, y_test = data
    n = X_test.shape[0]
    mse = mse_comp(y_test - (X_test @ param))
    return(mse)

def assum(gamma,r2,sigma2=1):
    mse = np.zeros((len(r2),len(gamma)))
    param_norm = np.zeros((len(r2),len(gamma)))
    
    for k in range(len(r2)):
        r = r2[k]
        for i in range(len(gamma)):
            g = gamma[i]

            if g <= 1:
                mse[k,i] = sigma2*g/(1-g)+sigma2
                param_norm[k,i] = r + sigma2*g/(1-g)
            else:
                mse[k,i] = r*(1-1/g) + sigma2/(g-1)+sigma2
                param_norm[k,i] = r/g + sigma2/(g-1)

    return(mse,param_norm)
        
def mse_comp(x):
    n = x.shape[0]
    return(1/n*np.sum(x**2))


def plotres(mse,param_norm,gamma,mse_assum,param_norm_assum,gamma_assum,r2):
    cutoff=10

    fig, (ax1,ax2) = plt.subplots(1,2)

    for k in range(mse.shape[0]):
        ax1.plot(gamma,mse[k,:],'*',color=f'C{k}')
        ax1.plot(gamma_assum,mse_assum[k,:],'-',color=f'C{k}')
    ax1.axvline(1,color="k",linestyle='--')
    ax1.set_xscale("log")
    ax1.set_ylim([0,cutoff])
    ax1.set_ylabel("MSE")
    ax1.set_xlabel(r'$\gamma$')

    for k in range(mse.shape[0]):
        ax2.plot(gamma,param_norm[k,:],'*',color=f'C{k}')
        ax2.plot(gamma_assum,param_norm_assum[k,:],'-',label=f'SNR={r2[k]}',color=f'C{k}')
    ax2.axvline(1,color="k",linestyle='--')
    ax2.set_xscale("log")
    ax2.set_ylim([0,cutoff])
    ax2.set_ylabel("2-norm")
    ax2.set_xlabel(r'$\gamma$')
    ax2.legend(loc='upper right',# bbox_to_anchor=(-0.125, 1.075),
               ncol=1, fancybox=True, shadow=False)

    plt.show()
    return 0 


def main(n=10):
    sigma2=1.
    r2vec = [1.,2.33,3.66,5.]
    
    gamma = np.logspace(-1,1,n)

    mse = np.zeros((len(r2vec),n))
    param_norm = np.zeros((len(r2vec),n))
    N = 300


    for k in range(len(r2vec)):
        r2=r2vec[k]
        for i in range(len(gamma)):
            p = int(gamma[i]*(2*N//3)) # training points
            #print(f'p: {p}')
           
            # generate data
            data=gen_data(N=N,p=p,r2=r2,sigma2=sigma2)
            # fit model
            param=fit_model(data)
            # compute norm of model
            param_norm[k,i] = norm(param)
            # evaluate the model on training data
            mse[k,i]=empirical_eval(param=param,data=data)

    # compute assumptots
    gamma2 = np.logspace(-1,1,n*10)
    mse_assum, param_norm_assum = assum(gamma=gamma2,r2=r2vec,sigma2=sigma2)

    ## plot result
    plotres(mse,param_norm,gamma,mse_assum,param_norm_assum,gamma2,r2vec)

    
    return(mse,gamma,param_norm,mse_assum,param_norm_assum,gamma2,r2vec)


if __name__ == "__main__":
    main()

    

