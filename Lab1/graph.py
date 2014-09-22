#!/usr/bin/python

import random
import csv
import math

from main import *

import matplotlib.pyplot as plt

def main():
	data = read_file('chips.txt')

	group0 = [x for x in data if x[2] == 0]
	group1 = [x for x in data if x[2] == 1]

	a = list(zip(*group0))
	b = list(zip(*group1))

	plt.plot(a[0], a[1], 'ro')
	plt.plot(b[0], b[1], 'rx')
	plt.show()

if __name__ == '__main__':
	main()