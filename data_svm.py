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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from time import time


def plotAccuracy(arr):
    """
        input - 2d array with parameter, value
    """
    acc_plot = pd.DataFrame(arr, columns=["x", "y"])
    
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 1, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.title("Accuracy vs. param")
    plt.tight_layout()
    plt.plot(acc_plot["x"], acc_plot["y"], label="cross-validation")
    plt.legend()

    

train_data = pd.read_csv("train.csv", header=None)
train_targets = pd.read_csv("trainLabels.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

scaler = StandardScaler()

train_data_scaled = pd.DataFrame(data=scaler.fit_transform(train_data))
targets = np.ravel(train_targets)
test_data_scaled = scaler.transform(test_data)

print("train size:", train_data_scaled.shape)
print("test size:", test_data_scaled.shape)

cv = KFold(n_splits=8, shuffle=True)
clf = SVC()
###get best paarams

grid = [
#  {'C': np.power(10.0, np.arange(-3, 3)), 'kernel': ['linear']},
  {'C': np.power(10.0, np.arange(-3, 3)), 'gamma': np.power(10.0, np.arange(-5, 3)), 'kernel': ['rbf']}
]

gs = GridSearchCV(clf, grid, scoring='accuracy', refit=False, cv=cv, return_train_score=False)

t0 = time()
print("Cross validation")
gs.fit(train_data_scaled, targets)
print("done in %.2fs" % (time() - t0))

cv_results = pd.DataFrame(gs.cv_results_)
cv_results.sort_values(by=['mean_test_score'], ascending=False, inplace=True)


###train and predict
bestC = cv_results['param_C'].iloc[0]
bestGamma = cv_results['param_gamma'].iloc[0]
bestKernel = cv_results['param_kernel'].iloc[0]
print("best params: C={}, gamma={}, kernel={}".format(bestC, bestGamma, bestKernel))
print("SVM Accuracy: ", cv_results['mean_test_score'].iloc[0])
clf = SVC(C=bestC, gamma=bestGamma, kernel=bestKernel)
clf.fit(train_data_scaled, train_targets)
predictions = clf.predict(test_data_scaled)

e = pd.DataFrame.from_records(data=enumerate(predictions),
                 columns=["Id", "Solution"])
e["Id"] = e["Id"] + 1

e.to_csv('submission.csv', index=False)
