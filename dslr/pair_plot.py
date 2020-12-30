#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("dataset_train.csv")

sns.pairplot(data, corner=True, hue="Hogwarts House")
plt.show()