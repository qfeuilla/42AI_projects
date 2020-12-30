#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import save
import pickle
import warnings

try:
	arg = sys.argv[1]
	if (len(sys.argv) > 3):
		raise Exception("aaaaaa")
	data = pd.read_csv(arg)
except:
	print("Please input one valid data file")
	exit()

data = data.fillna(data.mean())

try:
	arg = sys.argv[2]
	if (len(sys.argv) > 3):
		raise Exception("aaaaaa")
	f = open(arg, "rb")
	model = pickle.load(f)
except:
	print("Please input a valid model data")
	exit()

def normalize(df, mins, maxs):
	# return df normalized between -1 and 1
	result = df.copy()
	for i, feature_name in enumerate(df.columns):
		max_value = maxs[i]
		min_value = mins[i]
		result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
	return result * 2  + 1

X = normalize(data[model["col_X"]], model["mins"], model["maxs"]).to_numpy(dtype=np.float)

def sigmoid(X):
	return 1 / (1 + np.exp(-X))

def softmax(X):
	s_exp = np.sum(np.exp(X))
	return [np.exp(x) for x in X] / s_exp

def predict(weights, biases, x, act="softmax", one=False):
	if (one):
		z = np.dot(weights.T, x) + biases
	else:
		z = np.array([np.dot(w.T, x) + b for w, b in zip(weights, biases)])
	if (act == "sigmoid"):
		return sigmoid(z)
	if (act == "softmax"):
		return softmax(z)
	return z

predictions = []

for x in X:
	predictions.append(np.argmax(predict(model["weights"], model["biases"], x)))

predictions = [model["classes"][y] for y in predictions]
df = pd.DataFrame(data={'Index': range(len(predictions)), 'Hogwarts House': predictions})

df.to_csv('house.csv', index=False)
print(df.head())