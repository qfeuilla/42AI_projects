#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("dataset_train.csv")
data = data.fillna(data.mean())

def normalize(df):
   result = df.copy()
   for feature_name in df.columns:
      max_value = df[feature_name].max()
      min_value = df[feature_name].min()
      result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
   return result

Ravenclaw = normalize(data[(data["Hogwarts House"] == "Ravenclaw")].iloc[:, 7:])
Slytherin = normalize(data[(data["Hogwarts House"] == "Slytherin")].iloc[:, 7:])
Gryffindor = normalize(data[(data["Hogwarts House"] == "Gryffindor")].iloc[:, 7:])
Hufflepuff = normalize(data[(data["Hogwarts House"] == "Hufflepuff")].iloc[:, 7:])

labels = Ravenclaw.keys()
R_means = [np.mean(Ravenclaw[k]) for k in Ravenclaw.keys()]
S_means = [np.mean(Slytherin[k]) for k in Slytherin.keys()]
G_means = [np.mean(Gryffindor[k]) for k in Gryffindor.keys()]
H_means = [np.mean(Hufflepuff[k]) for k in Hufflepuff.keys()]

x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, R_means, width, label='Ravenclaw')
rects2 = ax.bar(x - width - width/2, S_means, width, label='Slytherin')
rects3 = ax.bar(x + width + width/2, G_means, width, label='Gryffindor')
rects4 = ax.bar(x + width/2, H_means, width, label='Hufflepuff')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized mean Scores')
ax.set_title('Scores by Hogwarts house')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()