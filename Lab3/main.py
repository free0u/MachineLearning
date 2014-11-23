#!/usr/bin/python

import random
import csv
import math
import matplotlib.pyplot as plt

def read_file(name):
	with open(name, 'r') as f:
		data = csv.reader(f, delimiter=',')
		ret = list(map(lambda x : [float(x[0]), float(x[1]), int(x[2])], data))
		return ret


def normalize(data, ind):
	arr = list(zip(*data))[ind]

	left = min(arr)
	right = max(arr)

	for row in data:
		v = row[ind]
		v = (v - left) / (right - left)
		row[ind] = v

def F1(out, cond):
	assert len(out) == len(cond)

	tp = fp = fn = 0
	for i in range(len(out)):
		o = out[i]
		c = cond[i]

		if o == 1 and c == 1:
			tp += 1
		
		if o == 1 and c == -1:
			fp += 1
		
		if o == -1 and c == 1:
			fn += 1

	# precision = float(tp) / (tp + fp)
	# recall = float(tp) / (tp + fn)

	# f1 = 2 * (precision) * (recall) / (precision + recall)
	
	f1 = 2.0 * tp / (2.0 * tp + fp + fn)

	return f1


def find_best_C(tol, max_passes, data):
	N_iter = 5
	N_train = int(0.8 * len(data))
	# C = [2.0 ** (i) for i in range(3, -10, -1)]
	C = []
	st = 2.5
	while st < 4:
		C.append(st)
		st += 0.05
	
	# C = range(10)
	print(C)

	f1s = []
	for c in C:
		print(c)
		f1_mean = 0
		for i in range(N_iter):
			random.shuffle(data)
			train = data[:N_train]
			test = data[N_train:]

			X_train = [[i[0], i[1]] for i in train]
			Y_train = [i[2] for i in train]
			X_test = [[i[0], i[1]] for i in test]
			Y_test = [i[2] for i in test]


			a, b, w = SVO(c, tol, max_passes, X_train, Y_train, inner_product)
			predicted = [classifier(w, b, i) for i in X_test]

			f1 = F1(predicted, Y_test)
			f1_mean += f1
		f1_mean = float(f1_mean) / N_iter
		print('f1_mean =', f1_mean)
		f1s.append((f1_mean, c))
	f1s.sort()

	print("c f1")
	for (f, c) in f1s:
		print(c, f)

	return f1s[-1][1]


def find_best_C_kernel(tol, max_passes, data, wrapper):
	N_iter = 5
	N_train = int(0.8 * len(data))
	# C = [2.0 ** (i) for i in range(3, -10, -1)]
	C = []
	st = 2.5
	while st < 4:
		C.append(st)
		st += 0.05
	
	# C = range(10)
	PHI = [2.0 ** (i) for i in range(3, -3, -1)]
	C = [2.0 ** (i) for i in range(3, -3, -1)]
	print(C, PHI)

	f1s = []
	for phi in PHI:
		for c in C:
			print(c, phi)
			f1_mean = 0
			for i in range(N_iter):
				random.shuffle(data)
				train = data[:N_train]
				test = data[N_train:]

				X_train = [[i[0], i[1]] for i in train]
				Y_train = [i[2] for i in train]
				X_test = [[i[0], i[1]] for i in test]
				Y_test = [i[2] for i in test]


				a, b, w = SVO(c, tol, max_passes, X_train, Y_train, wrapper(phi))
				# (x0, a, X, Y, b, kernel)
				predicted = [classifier_kernel(i, a, X_train, Y_train, b, wrapper(phi)) for i in X_test]

				f1 = F1(predicted, Y_test)
				f1_mean += f1
			f1_mean = float(f1_mean) / N_iter
			print('f1_mean =', f1_mean)
			f1s.append((f1_mean, (c, phi)))
	f1s.sort()

	print("f1 c phi")
	for i in f1s:
		print(i)

	return f1s[-1][1]


def inner_product(x1, x2):
	return x1[0] * x2[0] + x1[1] * x2[1]

def calc_f(x0, a, X, Y, b, kernel):
	m = len(X)
	ret = 0
	for i in range(m):
		ret += a[i] * Y[i] * kernel(X[i], x0)
	ret = ret + b
	return ret

def SVO(C, tol, max_passes, X, Y, kernel):
	assert len(X) == len(Y)
	m = len(X)

	a = [0.0] * m
	b = 0.0
	passes = 0
	while passes < max_passes:
		num_changed_alphas = 0
		for i in range(m):
			# print('i =', i)
			# calc E
			Ei = calc_f(X[i], a, X, Y, b, kernel) - Y[i]
			if (Y[i] * Ei < -tol and a[i] < C) or (Y[i] * Ei > tol and a[i] > 0):
				j = i
				while j == i:
					j = random.randint(0, m - 1)
				Ej = calc_f(X[j], a, X, Y, b, kernel) - Y[j]
				ai_old = a[i]
				aj_old = a[j]
				if Y[i] != Y[j]: # 0 C
					L = max(0, a[j] - a[i])
					H = min(C, C + a[j] - a[i])
				else: # 0 0
					L = max(0, a[i] + a[j] - C)
					H = min(C, a[i] + a[j])
				if abs(L - H) < 1e-5: # TODO
					continue
				n = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j])
				if n >= 0:
					continue
				a[j] = a[j] - (Y[j] * (Ei - Ej)) / n
				if a[j] > H:
					a[j] = H
				elif L <= a[j] <= H:
					pass
				else:
					a[j] = L
				if abs(a[j] - aj_old) < 1e-5:
					continue
				a[i] = a[i] + Y[i] * Y[j] * (aj_old - a[j])
				b1 = b - Ei - Y[i] * (a[i] - ai_old) * kernel(X[i], X[i]) - \
					Y[j] * (a[j] - aj_old) * kernel(X[i], X[j])
				b2 = b - Ej - Y[i] * (a[i] - ai_old) * kernel(X[i], X[j]) - \
					Y[j] * (a[j] - aj_old) * kernel(X[j], X[j])
				if 0 < a[i] < C:
					b = b1
				elif 0 < a[j] < C:
					b = b2
				else:
					b = (b1 + b2) / 2
				num_changed_alphas += 1
		if num_changed_alphas == 0:
			passes += 1
		else:
			passes = 0
	w = [0, 0]
	for i in range(m):
		w[0] += a[i] * Y[i] * X[i][0]
		w[1] += a[i] * Y[i] * X[i][1]
	return a, b, w

def classifier(w, b, x):
	t = w[0] * x[0] + w[1] * x[1] + b
	return -1 if t < 0 else 1

def classifier_kernel(x0, a, X, Y, b, kernel):
	m = len(X)
	ret = 0
	for i in range(m):
		ret += a[i] * Y[i] * kernel(X[i], x0)
	ret = ret + b
	return -1 if ret < 0 else 1

def main():
	random.seed(0)

	data = read_file('data.txt')
	for i in data:
		i[2] = -1 if i[2] == 0 else 1

	normalize(data, 0)
	normalize(data, 1)

	random.shuffle(data)
	N_train = int(0.8 * len(data))

	train = data[:N_train]
	test = data[N_train:]

	tol = 0.001
	max_passes = 10
	# C = find_best_C(tol, max_passes, train)
	C = 0.4

	X_train = [[i[0], i[1]] for i in train]
	Y_train = [i[2] for i in train]
	X_test = [[i[0], i[1]] for i in test]
	Y_test = [i[2] for i in test]

	a, b, w = SVO(C, tol, max_passes, X_train, Y_train, inner_product)
	# print('w =', w, 'b = ', b)
	c1 = -b / w[1]
	c2 = -w[0] / w[1]
	print("y = %f + %f * x" % (c1, c2))

	predicted = [classifier(w, b, i) for i in X_test]

	f1 = F1(predicted, Y_test)
	print('f1 =', f1)

def rbf(x1, x2, a):
	d = (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2
	return math.e ** (d / (-2.0 * a * a))

def rbf_wrapper(a):
	return lambda x,y : rbf(x, y, a)

def modify_data(data):
	for i in data:
		x = i[0]
		y = i[1]
		x -= 0.5
		y -= 0.5
		r = math.sqrt(x * x + y * y)
		i[0] = r
		i[1] = y


def main2():
	random.seed(0)

	data = read_file('data2.txt')
	for i in data:
		i[2] = -1 if i[2] == 0 else 1

	normalize(data, 0)
	normalize(data, 1)

	# modify_data(data)

	# group0 = [x for x in data if x[2] == -1]
	# group1 = [x for x in data if x[2] == 1]

	# a = list(zip(*group0))
	# b = list(zip(*group1))

	# plt.plot(a[0], a[1], 'ro')
	# plt.plot(b[0], b[1], 'rx')


	# plt.show()

	random.shuffle(data)
	N_train = int(0.8 * len(data))

	train = data[:N_train]
	test = data[N_train:]

	tol = 0.001
	max_passes = 10
	# C, a_par = find_best_C_kernel(tol, max_passes, train, rbf_wrapper)
	# C = 3.2

	C, a_par = 8.0, 0.5

	X_train = [[i[0], i[1]] for i in train]
	Y_train = [i[2] for i in train]
	X_test = [[i[0], i[1]] for i in test]
	Y_test = [i[2] for i in test]

	# a_par = 2.0
	kernel = rbf_wrapper(a_par)
	a, b, w = SVO(C, tol, max_passes, X_train, Y_train, kernel)
	# print('w =', w, 'b = ', b)
	# c1 = -b / w[1]
	# c2 = -w[0] / w[1]
	# print("y = %f + %f * x" % (c1, c2))

	#(x0, a, X, Y, b, kernel)
	predicted = [classifier_kernel(i, a, X_train, Y_train, b, kernel) for i in X_test]

	f1 = F1(predicted, Y_test)
	print('f1 =', f1)
	return f1

if __name__ == '__main__':
	main()
	print('=================')
	main2()

