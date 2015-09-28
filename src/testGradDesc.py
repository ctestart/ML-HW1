import numpy as np
import unittest
import gradDesc as gd
import homework1 as hw1

class testGradDesc(unittest.TestCase):

	def test_init(self):
		obj = gd.gradDesc(1, 1, .01, 'X', 'X**2', False, 'curvefitting.txt', 0)

	def test_F(self):
		obj = gd.gradDesc(1, 1, .01, 'X', 'X**2', False, 'curvefitting.txt', 0)
		X = np.array([1,2,3,4])
		self.assertEqual(obj.F(X), 5)

	def test_grad(self):
		obj = gd.gradDesc(1, 1, .01, 'X', 'X**2', False, 'curvefitting.txt', 0)
		X = np.array([1,2,3,4])
		h = .000001
		true_val = obj.grad(X)
		approx_val = obj.grad_approx(X, h)
	
		self.assertAlmostEqual(approx_val[0], true_val[0])
		self.assertAlmostEqual(approx_val[1], true_val[1])

	def test_grad_SEE(self):
		data_file = 'curvefitting.txt'
		order = 3
		[X, Y] = hw1.getData(data_file)
		phi = hw1.designMatrix(X, order)
		weights = hw1.regressionFit(X, Y, phi)
		delta = .001
		print hw1.computeSEEGrad(X,Y, weights, order)
		print gd.gradient_approx_SEE(X, Y, weights, order, delta)

	def test_gaussian(self):
		return 0

	def test_step(self):
		return 0

	def test_conv_criteria(self):
		return 0	

	#def test_grad_approx(self):
	#	init_guess = input_array = np.array([1,2])
	#	obj = gd.gradDesc(init_guess, .1, .00001, 'X', 'X**2', True, 'curvefitting.txt', 0)	
	#	obj.grad_approx(init_guess, .001)

if __name__ == '__main__':
    unittest.main()