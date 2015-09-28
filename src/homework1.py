import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import numpy as np
import unittest
import homework1
import gradDesc 
from numpy.linalg import inv, norm

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
    print w.shape
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
    """ NEED TO TEST """
    phiT = phi.transpose()
    return (np.matrix(phiT)* np.matrix(phi)).getI() * np.matrix(phiT) * Y

def computeSEE(X,Y,weights,order):
    """Compute the Sum of Square Error function given a dataset (X,Y)"""
    """a weight vector and the order of the polynomial basis functions"""
    phi_matrix=designMatrix(X,order)
    (n,m)=weights.shape
    if(m==0):
        weights = np.array(weights).reshape([n,1])
    #(n,_) = Y.shape
    #SEE_new = 0
    #wT = np.array(weights).reshape([(order+1)]).flatten()
    #for i in xrange(n):
    #    phi = phi_matrix[i]
    #    print "IN THE SUB", (Y[i] - np.dot(wT,phi))
    #    SEE_new += (0.5)*((Y[i] - np.dot(wT, phi))**2)
    SSE=(0.5) * np.sum(np.square(Y-((weights.T*np.matrix(phi_matrix.transpose())).T)))
    #SEE_with_dot = (0.5)*np.sum(np.square(Y-((weights.T.dot(phi)))))
    #print "NEW RESULT", SEE_new
    #print "OLD RESULT", SSE    
    return SSE

def computeSEEGrad(X,Y, weights, order):
    """ Compute the gradient of the SEE function given a dataset (X,Y) """
    """ the weight vector and the order of the polynomial base functions """
    phi=designMatrix(X,order)
    n = len(weights)
    w = np.array(weights).reshape([n,1])
    SEEGrad = (w.T*np.matrix(phi.transpose())-Y.T)*np.matrix(phi)
    #SEEGrad_with_dot = ((weights.T).dot(phi) - (Y.T)).dot(phi)
    return np.array(SEEGrad).flatten()

def ridge_regression(phi_matrix, l, Y):
    """ Returns theta_hat, MLE of theta """
    (_,d) = phi_matrix.shape                  
    lambda_matrix = l * np.eye(d)
    phiT = phi_matrix.T
    return (inv(lambda_matrix + (phiT.dot(phi_matrix)))).dot(phiT.dot(Y))

def run_ridge_find_lambda():
    """For figure 1.4"""
    """Question 3.1"""
    # Do for a few orders : line search to find the lambda
    for i in xrange(5):
        order=4
        [X,Y] = getData('curvefitting.txt')
        phi_mat=designMatrix(X,order)
        if i==0:
            lam = 0
        else:
            lam = 1.0/(5.0*i)
        theta_hat = ridge_regression(phi_mat, lam, Y)
        Y_hat = phi_mat.dot(theta_hat) ## Y estimate
        #pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')
        # produce a plot of the values of the function 
        #pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
        #Yp = pl.dot(theta_hat, designMatrix(pts, order).T)
        #pl.plot(pts, Yp.tolist()[0])
        #pl.show()
        #print Y_hat
        print "lambda = ", lam, " ", norm(Y-Y_hat, 2)
    line_search_ridge_for_lambda(phi_mat, Y)
    return 0

def line_search_ridge_for_lambda(phi, Y):
    #old = np.array.1
    new = 1 
    alpha = .1
    (_,d) = phi.shape
    new = np.ones([d,1])
    old = np.zeros([d,1])
    theta_hat = ridge_regression(phi, old, Y)
    print phi.T.shape
    print Y.shape
    while(norm(new-old,2) > .001):         # while not converged
        old = new
        theta_hat = ridge_regression(phi, old, Y)
        #lam_mat = old * np.eye(d)
        p = -(1.0/(norm(old,2)))*(phi.T.dot(Y))
        new = old + alpha*p # theta_hat
        #print new
    print new


def model_selection():
    """ Run a series of tests fo figure out M and lambda """
    """ Question 3.2 """
    [X,Y] = getData('curvefitting.txt')
    phi_mat=designMatrix(X,M)
    theta_hat = ridge_regression(phi_mat, 1, Y)
    return 0


def do_regression(M):
    [X,Y] = getData('curvefitting.txt')
    regressionPlot(X, Y, M)
    Phi_matrix=designMatrix(X,M)
    regressionFit(X,Y,Phi_matrix)

if __name__ == '__main__':
    M = 9
    do_regression(9)
    #SSE=computeSEE(X,Y,weight_vector,9)
    #print ('Sum of Square Error')
    #print (SSE)
    #print ('Gradient of SSE')
    #SSEG=computeSEEGrad(X,Y,weight_vector,9)
    #print (SSEG)

    #print ridge_regression(Phi_matrix, 1, Y)



