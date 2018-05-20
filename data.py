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
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score, KFold


scaler = StandardScaler()

train_data = pd.read_csv("train.csv", header=None)
train_targets = pd.read_csv("trainLabels.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

train_data_scaled = pd.DataFrame(data=scaler.fit_transform(train_data))
test_data_scaled = scaler.transform(test_data)

print("train size:", train_data_scaled.shape)
print("test size:", test_data_scaled.shape)

kf = KFold(n_splits=5, shuffle=True)
clf = Perceptron()
cv = cross_val_score(clf, train_data_scaled, y=train_targets, cv=kf)
print("Perceptron Accuracy: ", cv.mean())

clf.fit(train_data_scaled, train_targets)
predictions = clf.predict(test_data_scaled)

e = pd.DataFrame.from_records(data=enumerate(predictions),
                 columns=["Id", "Solution"])
e["Id"] = e["Id"] + 1

e.to_csv('submission.csv', index=False)
