#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from progress.bar import IncrementalBar

def estimate_price(t0, t1, data):
    return t0 + t1 * data

# read the csv with the datas
datas = pd.read_csv("datas/data.csv")

print("data head 5 :")
print(datas.head(5))
print()

# extracting training datas
X = np.array(datas.iloc[:, 0])
Y = np.array(datas.iloc[:, 1])

# mean of our inputs and outputs
x_mean = np.mean(X)
y_mean = np.mean(Y)

# speed from which the algorithm "learn"
lr_scalert0 = 1e5
lr_scalert1 = 1e-5
lr = 1e-7

# max iterations for Linear Regression
maxiter = 5000

# m : number of training exemples
m = len(X)

bar = IncrementalBar("epochs : ", max=maxiter)

# initialize parameters
old_t0, old_t1 = [-42, -42]
t0, t1 = [0, 0]

# if the algo update only by threshold, stop the learning
thresholdt0 = 0.1
thresholdt1 = 1e-6

#loss history
errors = []

while maxiter and not (np.abs(t0 - old_t0) < thresholdt0 and np.abs(t1 - old_t1) < thresholdt1):
    maxiter -= 1

    # save old value of thetas
    old_t0, old_t1 = [t0, t1]
    
    #predict with actual params
    preds = estimate_price(old_t0, old_t1, X)

    # update parameters
    cost0 = lr * np.sum((preds - Y)) / m
    cost1 = lr * np.sum((preds - Y) * X) / m
    t0 -= lr_scalert0 * cost0
    t1 -= lr_scalert1 * cost1

    # update progress bar
    bar.next()
bar.finish()

# save wheights in file
utils.deleteContent("datas/model")
f = open("datas/model", "w+")
f.write(",".join([str(t0), str(t1)]))
f.close()

# final preds
preds = estimate_price(t0, t1, X)

plt.scatter(X, Y, color = "red")
plt.plot(X, preds, color = "green")
plt.show()

print("mean error =", np.mean(np.abs(Y - preds)))