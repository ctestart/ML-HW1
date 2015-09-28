from __future__ import print_function
import numpy as np
import homework1 as hw1
import sympy
from scipy.optimize import fmin_bfgs


def Finite_Diff(Fmx, X, h):
	'''Calculates the finite difference equivalent of the gradient of the function in X'''
	'''using spacing h'''
	n=len(X)
	ones=np.ones_like(X)
	F_xhr= Fmx(X+0.5*h*ones)
	F_xhl= Fmx( X-0.5*h*ones)
	return ((F_xhr- F_xhl)/h)

def QuadBowl(X):
	'''Quadratic bowl function without summing elements'''
	return (X.T*X)

def sumQuadBowl(X):
	'''Quadratic bowl function including sum of elements'''
	return (np.sum(X.T*X))

def numGradDescent(F,Fmx, X_0, step, threshold, spacing, iterations=0):
	'''looks for the minimum of function F using the gradient descent method starting '''
	''' in X_0, using the step for next X if the threshold in not obtained, evaluating '''
	'''the gradient numerically using the spacing in the Fmx single variable functions'''
	F_X0=F(X_0)
	gradF_X0=Finite_Diff(Fmx, X_0,spacing)
	print (gradF_X0)
	X_1=np.array(X_0-step*gradF_X0)
	F_X1=F(X_1)
	if abs(F_X0-F_X1)<threshold:
		print (abs(F_X0-F_X1))
		print ('\nDescent terminated\nCurrent function value: ', end='')
		print (F_X1)
		print ('Iterations: '+str(iterations+1)+'\t\t Function evaluations: '+str((iterations+1)*4))
		print ('Last Grad: ', end='')
		print (gradF_X0)
		return (F_X1)
	return numGradDescent(F,Fmx,X_1,step, threshold, spacing, (iterations+1))

if __name__ == '__main__':
	X=np.array([4,8])
	step= 0.25
	thresh=0.00001
	sp=0.5
	print (sumQuadBowl(X))
	print (Finite_Diff(QuadBowl,X,0.5))
	print (fmin_bfgs(sumQuadBowl, X))
	print ('\nGradient Descent')
	print ('Initial Guess'+ str(X)+'\t\t Step: '+str(step))
	print ('Threshold: '+str(thresh)+'\t\t Spacing: '+str(sp))
	numGradDescent(sumQuadBowl,QuadBowl ,X,step,thresh, sp)


