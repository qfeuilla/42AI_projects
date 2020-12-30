#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("dataset_train.csv")

sns.regplot(x="Defense Against the Dark Arts", y="Astronomy", data=data, scatter_kws={"alpha": 0.2}, fit_reg=False)
plt.show()