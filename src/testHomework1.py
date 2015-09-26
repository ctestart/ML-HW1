import unittest
import homework1 as hw1
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
        return 0

    def test_compare_derivative(self):
        """ Verify the gradient using the numerical derivative code """
        """ Part of question 2.2 """
        [X,Y] = hw1.getData('curvefitting.txt')
        hw1.regressionPlot(X, Y, 9)
        Phi_matrix=hw1.designMatrix(X,9)
        print (Phi_matrix)
        weight_vector=hw1.regressionFit(X,Y,Phi_matrix)
        print (weight_vector)
        SSE=hw1.computeSEE(X,Y,weight_vector,9)
        print ('Sum of Square Error')
        print (SSE)
        print ('Gradient of SSE')
        SSEG=hw1.computeSEEGrad(X,Y,weight_vector,9)
        hw1.computeSEEGrad(X,Y, weights, order)

if __name__ == '__main__':
    unittest.main()






