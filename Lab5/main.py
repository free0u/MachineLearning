#!/usr/bin/python
import csv
import math
import random


user_ids = set()
item_ids = set()
rates = {}
raw_rates = []
rates_valid = []
test_ids = []

u_mean = 0.0
item_mean = {}
user_mean = {}

pu = {}
qi = {}
bi = {}
bu = {}


def read_train():
	print('> read_train')
	with open('train.csv', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader)
		for row in reader:
			user, item, rate = map(int, row)

			raw_rates.append((user, item, rate))

			user_ids.add(user)
			item_ids.add(item)

			t = rates.get(user, {})
			t[item] = rate
			rates[user] = t

def read_valid():
	print('> read_valid')
	with open('validation.csv', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader)
		for row in reader:
			user, item, rate = map(int, row)
			rates_valid.append((user, item, rate))


def read_test_ids():
	print('> read_test_ids')
	with open('test-ids.csv', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader)
		for row in reader:
			id, user, item = map(int, row)
			test_ids.append((id, user, item))

def get_normal_vec(n):
	ret = [random.gauss(0.4, 0.1) for i in range(n)]
	return ret

def train(f, lam, phi):
	# print('> train')
	for i in item_ids:
		bi[i] = get_normal_vec(1)[0]
		qi[i] = get_normal_vec(f)
	for u in user_ids:
		bu[u] = get_normal_vec(1)[0]
		pu[u] = get_normal_vec(f)

	cnt_iter = 0
	old_rmse = 10 ** 9

	while True:
		cnt_iter += 1

		rmse = 0.0
		cnt_rates = 0

		for (user, item, rate) in raw_rates:
			cnt_rates += 1
			
			predicted = predict(user, item)
			
			error = rate - predicted
			rmse += error ** 2

			bu[user] += phi * (error - lam * bu[user])
			bi[item] += phi * (error - lam * bi[item])

			for i in range(f):
				tq = qi[item][i] 
				tp = pu[user][i]
				qi[item][i] = tq + phi * (error * tp + lam * tq)
				pu[user][i] = tp + phi * (error * tq + lam * tp)

		rmse /= cnt_rates
		print(cnt_iter, ':', rmse)

		if (abs(old_rmse - rmse) < 1e-6) or (cnt_iter > 15):
			break

		old_rmse = rmse

def dot_product(a, b):
	res = 0.0
	for i in range(len(a)):
		res += a[i] * b[i]
	return res

def predict(user, item):
	new_user = not user in user_ids
	new_item = not item in item_ids

	res = 0
	res += u_mean
	if not new_item:
		res += bi[item]
	if not new_user:
		res += bu[user]
	if not new_item and not new_user:
		res += dot_product(qi[item], pu[user])
	return res


def preprocess():
	print('> preprocess')
	read_train()
	read_valid()
	read_test_ids()

	global u_mean

	cnt_u = 0
	for user in rates:
		for item in rates[user]:
			rate = rates[user][item]
			
			# update u
			u_mean += rate
			cnt_u += 1

			# update user mean
			t = user_mean.get(user, (0, 0))
			t = (t[0] + rate, t[1] + 1)
			user_mean[user] = t

			# update item mean
			t = item_mean.get(item, (0, 0))
			t = (t[0] + rate, t[1] + 1)
			item_mean[item] = t

	u_mean /= cnt_u
	for i in user_mean:
		t = user_mean[i]
		user_mean[i] = t[0] / t[1]
	for i in item_mean:
		t = item_mean[i]
		item_mean[i] = t[0] / t[1]


def test_on_valid():
	rmse = 0.0
	for (user, item, rate) in rates_valid:
		predicted = predict(user, item)
		rmse += (rate - predicted) ** 2
	rmse /= len(rates_valid)
	rmse = math.sqrt(rmse)
	return rmse


def main():
	print('> main')
	preprocess()
	random.seed(0)

	# f = 5
	# lam = 0.00021207031250000002
	# phi = 0.008

	# lams = [lam + i * 0.00001 for i in range(-3, 4)]
	# phis = [phi + i * 0.0001 for i in range(-3, 4)]

	# best_p = (-1, -1)
	# best_v = 100000

	# cntt = 0
	# for lam in lams:
	# 	for phi in phis:
	# 		print("========== best:", best_p)
	# 		print(cntt)
	# 		cntt += 1
	# 		try:
	# 			print(f, lam, phi)
	# 			train(f, lam, phi)
	# 			rmse = test_on_valid()
	# 			print(rmse)
	# 			if rmse < best_v:
	# 				best_v = rmse
	# 				best_p = (lam, phi)
	# 			print("best: ", best_p)
	# 		except:
	# 			print("oops")

	# print("total best:", best_p)
	# return

	# train(f, lam, phi)
	# print('rmse =', test_on_valid())
	# return

	# lam = 0.02
	# for f in range(2, 8):
	# 	phi = 2
	# 	for i in range(15):
	# 		try:
	# 			print(f, lam, phi)
	# 			train(f, lam, phi)
	# 			rmse = test_on_valid()
	# 			print(rmse)
	# 		except:
	# 			print("oops")
	# 		phi /= 2


	# f = 2
	# phi = 0.00390625
	# lam = 2
	# for i in range(15):
	# 	try:
	# 		print(f, lam, phi)
	# 		train(f, lam, phi)
	# 		rmse = test_on_valid()
	# 		print(rmse)
	# 	except:
	# 		print("oops")
	# 	lam /= 2


    # found by something like grid search
	f = 100
	lam = 0.00017207031250000002
	phi = 0.0076
	train(f, lam, phi)

	with open('answer.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['id', 'rating'])
		for (id, user, item) in test_ids:
			rate = predict(user, item)
			if rate < 1:
				rate = 1
			if rate > 5:
				rate = 5
			writer.writerow([id, rate])

	print('rmse =', test_on_valid())


if __name__ == '__main__':
	main()