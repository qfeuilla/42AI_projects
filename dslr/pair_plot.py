#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("datas/dataset_train.csv")

sns.pairplot(data, corner=True, hue="Hogwarts House")
plt.show()