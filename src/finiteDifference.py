from __future__ import print_function
import numpy as np
import homework1 as hw1
import sympy
import math
from scipy.optimize import fmin_bfgs


def Finite_Diff(F, X, h):
	'''Calculates the finite difference equivalent of the gradient of the function in X'''
	'''using spacing h'''
	FDiff=np.zeros_like(X)
	for n in range(0, len (X)):
		direction_vector=np.zeros_like(X)
		direction_vector[n]=1
		F_xhr= F(X+0.5*h*direction_vector)
		F_xhl= F(X-0.5*h*direction_vector)
		print (str(F_xhr)+' '+str(F_xhl))
		FDiff[n]=(F_xhr- F_xhl)/h
	return (FDiff)

def QuadBowl(X):
	'''Quadratic bowl function without summing elements'''
	return (X.T*X)

def sumQuadBowl(X):
	'''Quadratic bowl function including sum of elements'''
	return (np.sum(X.T*X))

def sumQuadBowlGrad(X):
	'''The analytical gradient of the function'''
	Grad=np.zeros_like(X)
	Grad[0]=2*X[0]
	Grad[1]=2*X[1]
	return Grad

def nonConvexFunction(X):
	'''A non-convex function (3x^4-8x^3+5x^2+y^2) to test gradient descent'''
	x=X[0]
	y=0
	if len(X)>1:
		y=X[1]
	return (3*pow(x,4)-8*pow(x,3)+5*(x**2)+y**2)

def nonConvexFunctionGrad(X):
	'''The gradrient of the nonConvexFunction defined above'''
	Grad=np.zeros_like(X)
	Grad[0]=12*(X[0]**3)-24*(X[0]**2)+10*X[0]
	Grad[1]=2*X[1]
	return Grad


def numGradDescent(F, X_0, step, threshold, spacing, analytical=False, QuadBowl=True , iterations=0):
	'''looks for the minimum of function F using the gradient descent method starting '''
	''' in X_0, using the step for next X if the threshold in no attained, evaluating '''
	'''the gradient numerically '''
	F_X0=F(X_0)
	print ('F('+str(X_0)+')='+str(F_X0))
	if (analytical and QuadBowl):
		gradF_X0=sumQuadBowlGrad(X_0)
		print ('Analytical Grad: ',end=' ')
		print (gradF_X0)
	elif (analytical and not QuadBowl):
		gradF_X0=nonConvexFunctionGrad(X_0)
		print ('Analytical Grad: ',end=' ')
		print (gradF_X0)
	else:
		gradF_X0=Finite_Diff(F, X_0,spacing)
		print ('Finite Diff Grad: ',end=' ')
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
	return numGradDescent(F,X_1,step, threshold, spacing, analytical,QuadBowl, (iterations+1))

if __name__ == '__main__':
	X=np.array([0.0,0.0])
	step= 0.25
	thresh=0.001
	sp=0.5
	# print (sumQuadBowl(X))
	print ('Calculating the gradient of Quad Bowl in '+str(X))
	print ('Finite difference with spacing '+str(sp))
	print (Finite_Diff(sumQuadBowl,X,sp))
	print ('Anaytical Gradient:')
	print (sumQuadBowlGrad(X))
	print ('\nCalculating the gradient of non-convex function in '+str(X))
	print ('Finite difference with spacing '+str(sp))
	print (Finite_Diff(nonConvexFunction,X,sp))
	print ('Anaytical Gradient:')
	print (nonConvexFunctionGrad(X))
	# print (fmin_bfgs(sumQuadBowl, X))
	# print ('\nQuad Bowl Gradient Descent')
	# print ('Initial Guess'+ str(X)+'\t\t Step: '+str(step))
	# print ('Threshold: '+str(thresh)+'\t\t Spacing: '+str(sp))
	# numGradDescent(sumQuadBowl,X,step,thresh, sp, True, True)
	# print ('\nNon-Convex Function gradient descent')
	# Y=np.array([0,2])
	# print (Finite_Diff(nonConvexFunction,Y,sp))
	# print ('Analytical Grad: '+str(nonConvexFunctionGrad(Y)))
	# F_Y=nonConvexFunction(Y)
	# print ('X= '+str(Y)+'\t\t F(X)= '+str(F_Y))
	# print ('Initial Guess'+ str(Y)+'\t\t Step: '+str(step))
	# print ('Threshold: '+str(thresh)+'\t\t Spacing: '+str(sp))
	# numGradDescent(nonConvexFunction,Y,step,thresh, sp, True, False)
	


