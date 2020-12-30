#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import save
import pickle

try:
	arg = sys.argv[1]
	if (len(sys.argv) > 2):
		raise Exception("aaaaaa")
	data = pd.read_csv(arg)
except:
	print("Please input one valid file")
	exit()

# To apply logistic regression to a multiclass problem with M class, 
# we have to break the dataset, in M part to train M different model

# first let's preprocess all the datas

# replace nan with the mean value of each rows:
# idea ? : mean of each house instead
data = data.fillna(data.mean())

# Let's say that the first row of the dataset will be the Y value
# and let's use only the numerical values for the sake of simplicity
# (it shouyld be enought)

def get_classes():
	try:
		index = int(input("what is the index of the classes : "))
		classes = data.iloc[:, index].unique()
		print("The value to predict are : ", classes)
		okay = input("is it ok ? (*/n) : ")
		if okay == "n":
			get_classes()
		return classes, index
	except:
		print("please input a valid index")
		get_classes()

classes, index = get_classes()

n_class = len(classes)

def normalize(df):
	# return df normalized between -1 and 1
	result = df.copy()
	mins = []
	maxs = []
	for feature_name in df.columns:
		max_value = df[feature_name].max()
		min_value = df[feature_name].min()
		maxs.append(max_value)
		mins.append(min_value)
		result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
	return result * 2  + 1, mins, maxs

print("len of data :", len(data))

def get_X_Ys(data, n_class, index, classes, test_split=15):
	Ys = []

	# create split list for train_test split
	train_splt = [np.random.rand() > (test_split / 100) for _ in range(len(data))]
	test_splt = [not i for i in train_splt]

	#get X by taking all the numerical rows
	col_X = []
	for k in data.keys():
		if type(data[k][0]) == np.float64:
			col_X.append(k)
	print("numerical values are : ", col_X, "\n")
	X = data[col_X]
	X, mins, maxs = normalize(X)

	for n in range(n_class):
		# get Y for each class
		Y = [int(i) for i in data.iloc[:, index] == classes[n]]
		Ys.append(Y)
	X_train = X[train_splt].to_numpy()
	X_test = X[test_splt].to_numpy()
	Ys_train = [np.array([d for i, d in enumerate(Y) if train_splt[i]]) for Y in Ys]
	Ys_test = [np.array([d for i, d in enumerate(Y) if test_splt[i]]) for Y in Ys]
	return X_train, Ys_train, X_test, Ys_test, col_X, mins, maxs

def ask_splits():
	try:
		splt = int(input("\nHow many percent of the dataset do you want to use as test ?: "))
		if (splt > 90 or splt < 0):
			raise Exception('None')
		return splt
	except:
		print("Please input an int between 0-90")
		ask_splits()

splt = ask_splits()

X, Ys, X_test, Ys_test, col_X, mins, maxs = get_X_Ys(data, n_class, index, classes, test_split=splt)

print("len X :", len(X))
print("len Y :", len(Ys[0]))
print("len X_test :", len(X_test))
print("len Y_test :", len(Ys_test[0]))
print()


for i, Y in enumerate(Ys):
	print(classes[i], "are : ", sum(Y))

print("\nX is shape :", X.shape)
print()

def create_n_wb(n, input_size):
	return np.random.randn(n, input_size, 1), np.random.randn(n, 1)

weights, biases = create_n_wb(n_class, X.shape[1])

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

def accuracy(weights, biases, X, Ys):
	acc = []
	for i, x in enumerate(X):
		acc.append(np.argmax(predict(weights, biases, x, act="softmax")) == np.argmax([Y[i] for Y in Ys]))
	return np.mean(acc) * 100

print("accuracy of random weights : ", accuracy(weights, biases, X_test, Ys_test), "%")

def train_weights(X, Ys, weights, biases, epochs=100, lr=0.0001, X_test=None, Ys_test=None, verbos=True, verb_each=10, verb_test=False, verb_loss=False):
	losses_history = [[] for _ in range(len(Ys))]
	pbar = tqdm(range(epochs))
	for e in pbar:
		e_loss = [[] for _ in range(len(Ys))]
		# loop through each models
		for i, Y in enumerate(Ys):
			for x, y in zip(X, Y):
				a = predict(weights[i], biases[i], x, act="sigmoid", one=True)
				loss = - (y * np.log(a) + (1 - y) * np.log(1 - a))
				e_loss[i].append(loss)
				x = np.array([x]).T
				weights[i] -= lr * ((a - y) * x)
				biases[i] -= lr * (a - y)

		for c in range(len(Ys)):
			losses_history[c].append(np.mean(e_loss[c]))
		
		if (verbos and e%verb_each == 0):
			if (verb_loss):
				print()
				for c in range(len(Ys)):
					print("{}_loss:".format(classes[c]), np.mean(e_loss[c]))
			
			metrics = []
			metrics.append("train_acc: {}".format(accuracy(weights, biases, X, Ys)))
			if (verb_test):
				metrics.append("test_acc: {}".format(accuracy(weights, biases, X_test, Ys_test)))
			pbar.set_description(", ".join(metrics))
	return weights, biases, losses_history

# test train
weights, biases, losses_history = train_weights(X, Ys, weights, biases, epochs=50, lr=0.001)

for c in range(len(Ys)):
	plt.plot(losses_history[c])

model = {'weights': weights, 'biases': biases, 'classes': classes, 'col_X': col_X, 'mins': mins, 'maxs': maxs}
f = open('model.pkl', "wb")
pickle.dump(model, f)
f.close()

plt.legend(classes, loc='upper right')	
plt.show()

print(accuracy(weights, biases, X_test, Ys_test))