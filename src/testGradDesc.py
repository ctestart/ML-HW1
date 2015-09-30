import numpy as np
import unittest
import gradDesc as gd
import homework1 as hw1

class testGradDesc(unittest.TestCase):
	def test_init(self):
		obj = gd.gradDesc(1, 1, .01, False, 'curvefitting.txt', 0)

	def test_F(self):
		obj = gd.gradDesc(1, 1, .01, False, 'curvefitting.txt', 0)
		X = np.array([1,2,3,4])
		self.assertEqual(obj.F(X), 5)

	def test_grad(self):
		obj = gd.gradDesc(1, 1, .01, False, 'curvefitting.txt', 0)
		X = np.array([1,2,3,4])
		h = .000001
		true_val = obj.grad(X)
		approx_val = obj.grad_approx(X, h)
		self.assertAlmostEqual(approx_val[0], true_val[0])
		self.assertAlmostEqual(approx_val[1], true_val[1])

	def test_grad_SEE(self):
		data_file = 'curvefitting.txt'
		for order in xrange(10):
			[X, Y] = hw1.getData(data_file)
			phi = hw1.designMatrix(X, order)
			weights = hw1.regressionFit(X, Y, phi)
			delta = 1
			analytic = hw1.computeSEEGrad(X,Y, weights, order).flatten()
			approx = gd.gradient_approx_SEE(X, Y, weights, order, delta).flatten()
			for i in xrange(order):	
				self.assertAlmostEqual(analytic[i], approx[i])

			delta2 = .1
			analytic2 = hw1.computeSEEGrad(X,Y, weights, order).flatten()
			approx2 = gd.gradient_approx_SEE(X, Y, weights, order, delta2).flatten()
			for i in xrange(order):	
				self.assertAlmostEqual(analytic2[i], approx2[i])

			delta3 = .01
			analytic3 = hw1.computeSEEGrad(X,Y, weights, order).flatten()
			approx3 = gd.gradient_approx_SEE(X, Y, weights, order, delta3).flatten()
			for i in xrange(order):	
				self.assertAlmostEqual(analytic3[i], approx3[i])

			delta4 = .001
			analytic4 = hw1.computeSEEGrad(X,Y, weights, order).flatten()
			approx4 = gd.gradient_approx_SEE(X, Y, weights, order, delta4).flatten()
			for i in xrange(order):	
				self.assertAlmostEqual(analytic4[i], approx4[i])

	def test_gaussian(self):
		return 0

	def test_step(self):
		return 0

	def test_conv_criteria(self):
		return 0	

	def test_grad_descent(self):
		M = 3
		#x = np.ones(M+1)
		x = np.array([.3, 8, -20, 17])
		des_obj = gd.gradDesc(x, .2, .01, True, 'curvefitting.txt', M)
		answer = des_obj.grad_descent(True)
		print "Found root at ", answer

if __name__ == '__main__':
    unittest.main()