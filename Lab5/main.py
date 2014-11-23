#!/usr/bin/python
import csv
import math


user_ids = set()
item_ids = set()
rates = {}
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


def train(f, lam, phi):
	# print('> train')
	for i in item_ids:
		bi[i] = 0.0
		qi[i] = [0.01] * f
	for u in user_ids:
		bu[u] = 0.0
		pu[u] = [0.01] * f

	cnt = 0
	old_measure = 10 ** 9
	while True:
		cnt += 1
		# if cnt > 1:
		# 	break

		measure = 0.0
		cnt_rates = 0
		for user in rates:
			items = rates[user]
			for item in items:
				rate = items[item]
				cnt_rates += 1
				
				predicted = predict(user, item)
				
				error = rate - predicted
				measure += error ** 2
				bu[user] += phi * (error - lam * bu[user])
				bi[item] += phi * (error - lam * bi[item])

				for i in range(f):
					qi[item][i] += phi * (error * pu[user][i]+ lam * qi[item][i])
					pu[user][i] += phi * (error * qi[item][i] + lam * pu[user][i])

		measure /= cnt_rates
		print(cnt, ':', measure)

		if old_measure < measure or cnt > 30:
			break

		old_measure = measure

def dot_product(a, b):
	assert len(a) == len(b)
	res = 0.0
	for (x, y) in zip(a, b):
		res += x * y
	return res

# def mul_vec_scalar(a, x):
# 	res = a[:]
# 	for (i, v) in enumerate(res):
# 		res[i] = v * x
# 	return res

# def add_vec_vec(a, b):
# 	res = a[:]
# 	for (i, v) in enumerate(res):
# 		res[i] = a[i] + b[i]
# 	return res

# def sub_vec_vec(a, b):
# 	res = a[:]
# 	for (i, v) in enumerate(res):
# 		res[i] = a[i] - b[i]
# 	return res


def predict(user, item):
	new_user = not user in user_ids
	new_item = not item in item_ids
	if new_user and new_item:
		return u_mean
	if new_user and not new_item:
		return item_mean[item]
	if not new_user and new_item:
		return user_mean[user]

	res = 0
	res += u_mean
	res += bi[item]
	res += bu[user]
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

	 # train(f, lam, phi):
	train(5, 0.0001220703125, 0.005)
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


	with open('answer.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['id', 'rating'])
		for (id, user, item) in test_ids:
			rate = predict(user, item)
			writer.writerow([id, rate])

	print('rmse =', test_on_valid())


if __name__ == '__main__':
	main()