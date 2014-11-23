#!/usr/bin/python

import random
import csv
import math
from numpy import matrix as M
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def read_file(name):
	with open(name, 'r') as f:
		data = csv.reader(f, delimiter=',')
		ret = list(map(lambda x : [float(x[0]), float(x[1]), float(x[2])], data))
		return ret


def normalize(data, ind):
	arr = list(zip(*data))[ind]

	right = max(arr)

	for row in data:
		v = row[ind]
		row[ind] = v / right

	# left = min(arr)
	# right = max(arr)

	# for row in data:
	# 	v = row[ind]
	# 	v = (v - left) / (right - left)
	# 	row[ind] = v


def getf(w0, w1, w2):
	return lambda x: w0 * x[0] + w1 * x[1] + w2 * x[2]

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

def main():
	random.seed(0)

	data = read_file('prices.txt')

	normalize(data, 0)
	normalize(data, 1)
	normalize(data, 2)

	A = [[1, x1, x2] for [x1, x2, x3] in data]
	y = [x3 for [x1, x2, x3] in data]


	mA = M(A)
	my = M.getT(M(y))

	w = M.getI(M.getT(mA) * mA) * (M.getT(mA) * my)

	w = w.getA1()
	f = getf(w[0], w[1], w[2])

	sse = 0.0

	for (i, p) in zip(A, y):
		t = (f(i) - p) ** 2
		sse += t

	mse = sse / len(data)
	rmse = math.sqrt(mse)
	print('sse =', sse)
	print('mse =', mse)
	print('rmse =', rmse)


	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	n = 100
	xs = list(zip(*data))[0]
	ys = list(zip(*data))[1]
	zs = list(zip(*data))[2]
	ax.scatter(xs, ys, zs, c='r', marker='o')

	ax.set_xlabel('S')
	ax.set_ylabel('rooms')
	ax.set_zlabel('price')

	normal = np.array([w[1], w[2], -1])
	point = np.array([0, 0, w[0]])
	d = -np.sum(point * normal)
	xx, yy = np.meshgrid([0.0, 1.0], [0.0, 1.0])
	z = (-normal[0] * xx - normal[1] * yy - d) * 1./normal[2]

	ax.plot_surface(xx,yy,z, color='grey')

	plt.show()


	# w = (A^T * A)^-1 * (A^T * y)



if __name__ == '__main__':
	main()






