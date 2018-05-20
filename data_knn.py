#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:19:20 2018

@author: andribas
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from matplotlib import pyplot as plt


scaler = StandardScaler()

train_data = pd.read_csv("train.csv", header=None)
train_targets = pd.read_csv("trainLabels.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

train_data_scaled = pd.DataFrame(data=scaler.fit_transform(train_data))
test_data_scaled = scaler.transform(test_data)

print("train size:", train_data_scaled.shape)
print("test size:", test_data_scaled.shape)

kf = KFold(n_splits=8, shuffle=True)

accuracy_k = []

for k in range(3, 101, 1):
    neigh = KNeighborsClassifier(n_neighbors=k)
    cv = cross_val_score(neigh, train_data_scaled, y=np.ravel(train_targets), cv=kf)
    accuracy_k.append([k, cv.mean()])

acc_k = sorted(accuracy_k,
                     key = lambda x : x[1],
                     reverse = True)

acc_plot = pd.DataFrame(accuracy_k, columns=["k", "mean"])

plt.figure(figsize=(8, 6))
plt.subplot(1, 1, 1)
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.title("Accuracy vs. k")
plt.tight_layout()
plt.plot(acc_plot["k"], acc_plot["mean"], label="cross-validation")
plt.legend()



print("KNN Accuracy: ", acc_k[0])
n = 72#acc_k[0][0]
clf = KNeighborsClassifier(n_neighbors=n)
clf.fit(train_data_scaled, np.ravel(train_targets))
predictions = clf.predict(test_data_scaled)

e = pd.DataFrame.from_records(data=enumerate(predictions),
                 columns=["Id", "Solution"])
e["Id"] = e["Id"] + 1

e.to_csv('submission.csv', index=False)
