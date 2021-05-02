# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:52:03 2021

@author: kalaivanan
"""

from sklearn.svm import SVC
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def sci_svm():
    
    #X = np.array([[3,4],[1,4],[2,3],[6,-1],[7,-1],[5,-3]] )
    #y = np.array([-1,-1, -1, 1, 1 , 1 ])

    #X = np.array([[1,3],[2,3],[3,-3],[4,-3]])
    #y = np.array([-1,-1,1,1])
    
    #X=np.array([[2,1], [2,-1], [3,0]])
    #y=np.array([-1, -1, 1])

    X=np.array([[1,1], [5,1], [1,4], [1,3]])
    y=np.array([-1, -1, 1, 1])
   
    clf = SVC(C = 1e5, kernel = 'linear')
    clf.fit(X, y) 
    print(clf.predict([[10,0]]))
    print('w = ',clf.coef_)
    print(round(clf.coef_[0][0], 2), round(clf.coef_[0][1],2))
    print('b = ',clf.intercept_)
    print('Indices of support vectors = ', clf.support_)
    print('Support vectors = ', clf.support_vectors_)
    print('Number of support vectors for each class = ', clf.n_support_)
    print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
    plot(X, clf.coef_, clf.intercept_)
    
def geo_margin(y,w,b,x):
    return y*(w / np.linalg.norm(w)).T @ x + (b / np.linalg.norm(w)).flatten()

def functional_margin(y,w,b,x):
    return y*(w.T @ x + b).flatten()

def f(x, w, b, c=0):
    # given x1, return x2 such that [x1,x2] are on the line w.x + b = c
    return (-w[0] * x - b + c) / w[1]

def plot(X, w, b):
    plt.scatter(X[:, 0], X[:, 1], )
    # optimal hyperplane
    ax = plt.gca()
    xlim = ax.get_xlim()

    if round(w[0][1], 2) == 0:
        ylim = ax.get_ylim()
        yy = np.linspace(ylim[0], ylim[1], 20)
        xx = np.array([-1*b[0]/w[0][0]]* len(yy))
    elif round(w[0][0], 2) == 0:
        xlim = ax.get_xlim()
        xx = np.linspace(xlim[0], xlim[1], 20)
        yy = np.array([-1*b[0]/w[0][1]]* len(xx))
    else:
        a = -w[0][0] / w[0][1]
        xx = np.linspace(xlim[0], xlim[1])
        yy = a * xx - b[0] / w[0][1]

    print(xx)
    print(yy)    
    plt.plot(xx,yy)
