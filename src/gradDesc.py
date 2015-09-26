import numpy as np

class gradDesc():
	def __init__(self, x0, step_size, eps, verbose=False):
		""" Specify the initial guess, the step size and the convergence criterion"""
		""" Verbose prints debug messages for checking functions and things """
		self.first = x0
		self.step_size = step_size 	# eta
		self.eps = eps
		self.verbose = verbose

	def grad_descent(self):
		""" Run gradient descent on a scalar function """
		old = self.first
		new = self.step(old)
		num_steps = 0 								# To keep track of how many iterations
		while(not conv_criteria(old, new, self.eps)):
			old = new
			new = self.step(old)
			num_steps+=1
			if self.verbose:
				print "step ", num_steps, " old ", old, " new ", new
		return new

	def step(self, old):
		if self.verbose:
			print "     GRADIENT FOR STEP ",  gradient(old)
		new = old - self.step_size * gradient(old)
		return new

def gradient(x):
	""" Change this for the appropriate function """
	return np.array([2*x[0], 2*x[1]])

def gradient_approx(x, delta):
	""" Calculates the gradient using finite differences """
	delta_vect = delta * np.ones(len(x))
	diff = (1/(2*delta)) * (f(x+delta_vec) - f(x-delta_vec))
	return diff

def f(x):
	""" define this yourself """
	return x[0]*x[0] + x[1]*x[1]

def conv_criteria(current, previous, eps):
	""" Determines whether the algorithm has converged by the two-norm """
	diff = np.linalg.norm(current-previous, 2)
	if diff <= eps:
		return True
	else:
		return False	

if __name__ == '__main__':
	x = np.array([1,2])
	des_obj = gradDes(x, .2, .00001, True)
	answer = des_obj.grad_descent()
	print "Found root at ", answer
    