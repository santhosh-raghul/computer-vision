#!/usr/bin/env python3

import numpy as np

def print_polynomial(p,var='x'):
	n = p.shape[0]
	s = ''
	for i in range(n-1,0,-1):
		s = s+"{}"+f"{var}^{i} + "
	s = s + "{}"
	print(s.format(*p[::-1]))

def convlove_1d(x,h):

	x_l,h_l = x.shape[0],h.shape[0]
	if x_l > h_l:
		x,h = h,x
		x_l,h_l = h_l,x_l

	y = []
	for n in range(x_l+h_l-1):
		y_ = 0
		for k in range(x_l):
			if (n-k)>=0 and (n-k)<h_l:
				y_ += x[k]*h[n-k]
		y.append(y_)

	return np.array(y)

if __name__=="__main__":

	print("enter coefficients of the polynomials in reverse, separated by spaces\nfor example, 3x^3 + 7x^2 + 1 will be as follows:\n1 0 7 3\n")

	print("enter polynomial 1: ",end='')
	p1 = np.array([int(i) for i in input().split()])
	print("you entered : ",end='')
	print_polynomial(p1)

	print("\nenter polynomial 2: ",end='')
	p2 = np.array([int(i) for i in input().split()])
	print("you entered : ",end='')
	print_polynomial(p2)

	print("\nthe product is : ",end='')
	print_polynomial(convlove_1d(p1,p2))