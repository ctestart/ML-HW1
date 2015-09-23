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

def computeSEE(X,Y,weights,order):
    """Compute the Sum of Square Error function given a dataset (X,Y), a weight vector and the order of the polynomial basis functions"""
    phi=designMatrix(X,order)
    SSE=(0.5) * np.sum(np.square(Y-((weights.T*np.matrix(phi.transpose())).T)))
    return SSE

def computeSEEGrad(X,Y, weights, order):
    """Compute the gradient of the SEE function given a dataset (X,Y), the weight vector and the order of the polynomial base functions"""
    phi=designMatrix(X,order)
    SEEGrad= (weights.T*np.matrix(phi.transpose())-Y.T)*np.matrix(phi)
    return SEEGrad

if __name__ == '__main__':
    [X,Y] = getData('curvefitting.txt')
    regressionPlot(X, Y, 9)
    Phi_matrix=designMatrix(X,9)
    print (Phi_matrix)
    weight_vector=regressionFit(X,Y,Phi_matrix)
    print (weight_vector)
    SSE=computeSEE(X,Y,weight_vector,9)
    print ('Sum of Square Error')
    print (SSE)
    print ('Gradient of SSE')
    SSEG=computeSEEGrad(X,Y,weight_vector,9)
    print (SSEG)



