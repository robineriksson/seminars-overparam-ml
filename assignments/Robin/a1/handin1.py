##' Robin Eriksson 2021-09-14
##
##
## Dependencies:
##    sklearn    (0.24.2)
##    scipy      (1.7.1)
##    numpy      (1.19.5)
##    matplotlib (3.4.3)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from scipy.linalg import lstsq, norm
import numpy as np
import matplotlib.pyplot as plt



def getdata():
    """
    getdata collects the data and constructs the train & test split

    output
        X_train: numpy training observables
        X_tests: numpy testing observables
        y_train: numpy training respons
        y_train: numpy testing respons

    """
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=0)
    return(X_train, X_test, y_train, y_test)

def train_model(gamma=1,seed=42,m=0.1,data=None):
    """
    Main model training method.
    1) fetch the data
    2) construct the RBF features
    3) min-norm LSQ on training
    4) min-norm LSQ on test
    5) compute the norm error and the norm of the trained parameters

    input
        gamma (int): kernel width
        seed (int): random seed
        m (float): m*training points -> number of features
        data (list): list of data [0] X_train, [1] X_test, [2] y_train, [3] y_test
    output
       res_train (float): training error [||fit - data||]
       res_test (float): test error
       res_train (float): training parameters norm
    """
    if data is None:
        X_train, X_test, y_train, y_test = getdata()
    else:
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
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



def plots(train,test,train_norm,m,gamma):
    """
    plots generates the pyplots that visualize the results

    input
       train (np float): training error
       test (np float): testing error
       train_norm (np float): training parameter norm
       m (np float): number of features / number of training points
       gamma (list): the different gammas used


    """

    cutoff = 1_000

    label = [r'g: ' + str(x)  for x in gamma]

    ## 1. train lsq error
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    dim = train.shape

    ax1.plot(m,train,label=label)
    ax1.set_xscale("log")
    ax1.axvline(1,color="k",linestyle='--',label="interpolation point")
    ax1.set_ylim([0,cutoff])
    ax1.set_ylabel("2-norm")

    ## 2. test lsq error
    ax2.plot(m,test,label=label)
    ax2.axvline(1,color="k",linestyle='--')
    ax2.set_xscale("log")
    ax2.set_ylim([0,cutoff])


    ax2.set_xlabel("number of features / number of training points")


    ax2.label_outer()
    ax3.label_outer()

    ## 3. parameter l_2 norm ||theta||_2
    ax3.plot(m,train_norm,label=label)

    ## cleanup 2
    ax3.axvline(1,color="k",linestyle='--',label="interpolation point")
    ax3.set_ylim([0,cutoff])

    ax3.set_xscale("log")

    ax3.legend(loc='upper center', bbox_to_anchor=(-0.75, 1.1),
               ncol=3, fancybox=True, shadow=True)
    plt.show()
    return



def experiment(points=100,gamma=[1],seed=42):
    m = np.logspace(-1,1,points)
    train_error = np.zeros((points,len(gamma)))
    test_error = np.zeros((points,len(gamma)))
    train_norm = np.zeros((points,len(gamma)))

    data = getdata()
    for i in range(len(m)):
        print(i)
        for j in range(len(gamma)):
            train_error[i,j], test_error[i,j],train_norm[i,j] = train_model(gamma=gamma[j],
                                                                  seed=seed,
                                                                  m=m[i],
                                                                  data=data)

    plots(train_error,test_error,train_norm,m,gamma)
    return(train_error,test_error,train_norm,m,gamma)

def main():
    experiment(points=100,gamma=[0.001,0.01,0.1,1],seed=42)

if __name__ == "__main__":
    main()
