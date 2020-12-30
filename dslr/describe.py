#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np

try:
	arg = sys.argv[1]
	if (len(sys.argv) > 2):
		raise Exception("aaaaaa")
	data = pd.read_csv(arg)
except:
	print("Please input one valid file")
	exit()

def percentile(d, p, cnt):
	rind = int(np.floor(cnt * p))
	return ((d[rind+1] + d[rind]) / 2)

all_features = [[" ", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]]
for k in data.keys():
	# if data is numerical, get datas
	if (type(data[k][0]) == np.float64) and len(all_features) < 5:
		d = np.array([i for i in data[k] if not np.isnan(i)])
		if len(d) == 0:
			continue
		one_feature = []
		one_feature.append(k)
		cnt = len(d)
		one_feature.append(str(cnt))
		mean = sum(d / cnt)
		one_feature.append(str(mean))
		std = np.sqrt(np.sum(np.square(d - mean)) / (cnt - 1))
		one_feature.append(str(std))
		d = np.array(sorted(d))
		one_feature.append(str(d[0]))
		one_feature.append(str(percentile(d, 0.25, cnt)))
		one_feature.append(str(percentile(d, 0.50, cnt)))
		one_feature.append(str(percentile(d, 0.75, cnt)))
		one_feature.append(str(d[-1]))
		all_features.append(one_feature)

space_between_col = 2

print("\n".join([" ".join([('%.{}s'.format(len(f[0]) + space_between_col))%(f[i].rjust(len(f[0]) + space_between_col)) if f[0] != " " else f[i].ljust(5 + space_between_col)  for f in all_features]) for i in range(len(all_features[0]))]))