##' Robin Eriksson 2021-09-14

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy.linalg import lstsq, norm
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import matplotlib.pyplot as plt

def train_model(gamma=1,seed=42,m=0.1):
    X_train, X_test, y_train, y_test = getdata()
    n_components = int(X_train.shape[0]*m)

    rbf_feature = RBFSampler(gamma=gamma,n_components=n_components,random_state=seed)

    ## train
    X_train_features = rbf_feature.fit_transform(X_train)
    param_train,_,_,_ = lstsq(X_train_features,y_train)
    res_train = norm((X_train_features @ param_train)-y_train)

    ## test
    X_test_features = rbf_feature.transform(X_test)
    res_test = norm((X_test_features @ param_train)-y_test)


    return(res_train,res_test,norm(param_train))


def getdata():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=0)
    return(X_train, X_test, y_train, y_test)


def plots(train,test,train_norm,m):
    ## 1. train lsq error
    plt.subplot(1,2,1)
    plt.plot(m,train,label="train")
    #plt.log("x")

    ## 2. test lsq error
    plt.plot(m,test,label="test")
    plt.xscale("log")
    plt.axvline(1,color="k",linestyle='--',label="interpolation point")
    plt.ylim([0,1_000])
    plt.ylabel("2-norm")
    plt.xlabel("num features / num training points")
    plt.legend()

    ## 3. parameter l_2 norm ||theta||_2
    plt.subplot(1,2,2)
    plt.plot(m,train_norm,color="r",label=r'$\theta$')
    plt.axvline(1,color="k",linestyle='--')
    plt.ylim([0,1_000])
    plt.xlabel("num features / num training points")

    plt.xscale("log")
    plt.legend()
    plt.show()
    return 0



def experiment(points=100,gamma=1,seed=42):
    m = np.logspace(-1,1,points)
    train_error = np.zeros(points)
    test_error = np.zeros(points)
    train_norm = np.zeros(points)

    for i in range(len(m)):
        print(i)
        train_error[i], test_error[i],train_norm[i] = train_model(gamma=gamma,seed=seed,m=m[i])

    plots(train_error,test_error,train_norm,m)
    return(train_error,test_error,train_norm,m)
