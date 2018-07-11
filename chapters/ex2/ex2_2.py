# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 12:23:55 2018

@author: N12667
"""

import numpy as np
from ml import mapFeature, mapFeature2
import pandas as pd
from matplotlib import pyplot as plt
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg

data2 = pd.read_csv('ex2data2.txt', delimiter=',', header=None, names=['Test 1', 'Test 2', 'Accepted'])

x1 = data2['Test 1']
x2 = data2['Test 2']
data2.insert(3, 'Ones', 1)
data2_original = data2.copy()

def plot_ex2(data2):
    positive = data2[data2['Accepted'].isin([1])]
    negative = data2[data2['Accepted'].isin([0])]
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')

def create_polynomial(data2, degree):
    for i in range(1, degree):
        for j in range(0, i):
            data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
    return data2

degree = 5
data2 = create_polynomial(data2, degree)  

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)
Lambda = 1

cost = costFunctionReg(theta2, X2, y2, Lambda )
grad = gradientFunctionReg(theta2, X2, y2, Lambda)