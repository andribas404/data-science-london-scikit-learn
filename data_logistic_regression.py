#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 11:52:17 2018

@author: andribas
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from time import time


class LogisticRegression:
    def __init__(self, k=0.01, C=1):
        self.w = None
        self.k = k
        self.C = C

    def fit(self, x, y):
        size = x.shape[1]
        #xx = x.assign(w0=np.ones(x.shape[0]))
        self.w = pd.Series(data=np.random.randn(size), name="w")

        for step in range(10000):
            wy = np.multiply(np.power(np.exp(np.multiply(np.matmul(x , self.w), y) * (-1)) + 1, -1) * (-1) + 1, y)
            wy = pd.DataFrame(wy)
            wy = pd.concat([wy * size], axis=1)
            wy = np.multiply(x, wy)
            wy = self.k * wy.mean()
            wnext = pd.DataFrame()
            wnext["w"] = self.w
            wnext = wnext.assign(y=list(wy))
            wnext["y"] = wnext["w"] * (1 - self.k * self.C) + wnext["y"]
            wnext["dist"] = np.power(wnext["y"] - wnext["w"], 2)
            #print(wnext)
            dist = np.power(wnext["dist"].sum(), 0.5)
            if dist < 1e-5:
                break
            self.w = wnext["y"]
    
        print("Gradient takes %d steps" % step)

    def predict(self, x):
        p = np.matmul(x , self.w)
        p = np.piecewise(p, [p < 0, p >= 0], [0, 1])
        return p.astype(int)
    
    def getROC(self, x, y):
        p = np.power(np.exp(np.matmul(x , self.w) * (-1)) + 1, -1)
        return roc_auc_score(y, p)

    def setParam(self, k=None, C=None):
        if k is not None:
            self.k = k
        if C is not None:
            self.C = C



train_data = pd.read_csv("train.csv", header=None)
train_targets = pd.read_csv("trainLabels.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

scaler = StandardScaler()

train_data_scaled = pd.DataFrame(data=scaler.fit_transform(train_data))
targets = np.ravel(train_targets)
test_data_scaled = scaler.transform(test_data)

print("train size:", train_data_scaled.shape)
print("test size:", test_data_scaled.shape)

clf = LogisticRegression(k=0.0001, C=10)
clf.fit(train_data_scaled, targets)
roc = clf.getROC(train_data_scaled, targets)

print("AUCROC = %.4f" % roc)

predictions = clf.predict(test_data_scaled)

e = pd.DataFrame.from_records(data=enumerate(predictions),
                 columns=["Id", "Solution"])
e["Id"] = e["Id"] + 1

e.to_csv('submission.csv', index=False)
