import unittest
import homework1 as hw1
import gradDesc as gd
import numpy as np

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions

class testHomework1(unittest.TestCase):

    def test_designMatrix(self):
        # X, order
        return 0


    def test_regressionFit(self):
        #(X, Y, phi):
        """ Compute the weight vector """
        """ w_ML = (phiT phi)^-1 phiT """
        return 0

    def test_computeSEE(self):
        order = 2
        [X,Y] = hw1.getData('curvefitting.txt')
        phi_matrix = hw1.designMatrix(X,order)
        weights = hw1.regressionFit(X,Y,phi_matrix)
        hw1.computeSEE(X,Y,weights,order)

    def test_ridge_regression(self):
        return 0

    def test_run_ridge_find_lambda(self):
        print hw1.run_ridge_find_lambda()

    def test_compare_derivative(self):
        """ Verify the gradient using the numerical derivative code """
        """ Part of question 2.2 """
        order = 4
        [X,Y] = hw1.getData('curvefitting.txt')
        Phi_matrix=hw1.designMatrix(X,order)
        #print Phi_matrix.shape
        weight_vector=hw1.regressionFit(X,Y,Phi_matrix)
        SSEG=hw1.computeSEEGrad(X,Y,weight_vector,order)
        approx = gd.gradient_approx_SEE(X, Y, weight_vector, order, 1e-10)
        true_deriv = hw1.computeSEEGrad(X,Y, weight_vector, order)

        print "COMPARE approx", approx
        print "TRUE ", true_deriv
        # TODO- Smaller test cases

if __name__ == '__main__':
    unittest.main()






