#!/usr/bin/python

import random
import csv
import math


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


def distance(a, b):
	x1, y1 = a[0], a[1]
	x2, y2 = b[0], b[1]
	return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def KNN(train, test, k):
	res = []
	for item_test in test:
		dist = []
		for (i, item_train) in enumerate(train):
			dist.append((distance(item_test, item_train), i))
		dist.sort()

		cnt = [0, 0]
		for i in range(k):
			item_train = train[dist[i][1]]
			cnt[item_train[2]] += 1

		res.append(0 if cnt[0] >= cnt[1] else 1)
	return res


def F1(out, cond):
	assert len(out) == len(cond)

	tp = fp = fn = 0
	for i in range(len(out)):
		o = out[i]
		c = cond[i]

		if o == 1 and c == 1:
			tp += 1
		
		if o == 1 and c == 0:
			fp += 1
		
		if o == 0 and c == 1:
			fn += 1

	precision = float(tp) / (tp + fp)
	recall = float(tp) / (tp + fn)

	f1 = 2 * (precision) * (recall) / (precision + recall)
	return f1


def find_best_k(data):
	N_iter = 10
	N_k = 20
	N_train = int(0.8 * len(data))

	f1s = []
	for k in range(1, N_k):
		f1_mean = 0
		for i in range(N_iter):
			t = data[:]
			random.shuffle(t)
			train = t[:N_train]
			test = t[N_train:]

			groups_knn = KNN(train, test, k)
			groups_true = list(map(lambda x: x[2], test))

			f1_mean += F1(groups_knn, groups_true)
		f1_mean = float(f1_mean) / N_iter
		f1s.append((f1_mean, k))
	f1s.sort()

	print("k f1")
	for (f, k) in f1s:
		print(k, f)

	return f1s[-1][1]


def main():
	random.seed(0)

	data = read_file('chips.txt')

	normalize(data, 0)
	normalize(data, 1)

	random.shuffle(data)
	N_train = int(0.8 * len(data))

	train = data[:N_train]
	test = data[N_train:]

	k = find_best_k(train)

	groups_knn = KNN(train, test, k)
	groups_true = list(map(lambda x: x[2], test))

	f1 = F1(groups_knn, groups_true)

	print()
	print("k =", k)
	print("F1 =", f1)

if __name__ == '__main__':
	main()