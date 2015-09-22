import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import numpy as np

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    w = regressionFit(X, Y, phi)

    print 'w', w
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])
    pl.show()

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('curvefitting.txt')

def regressTrainData():
    return getData('regress_train.txt')

def regressValidateData():
    return getData('regress_validate.txt')

def designMatrix(X, order):
    n = len(X)
    phi = np.empty([n, (order+1)])
    XX = np.asarray(X).reshape(n)
    for i in xrange(order+1):
        phi[:,i] = XX ** i
    return phi

def regressionFit(X, Y, phi):
    """ Compute the weight vector """
    """ w_ML = (phiT phi)^-1 phiT """
    phiT = phi.transpose()
    return (np.matrix(phiT)* np.matrix(phi)).getI() * np.matrix(phiT) * Y

if __name__ == '__main__':
    [X,Y] = getData('curvefitting.txt')
    regressionPlot(X, Y, 1)


