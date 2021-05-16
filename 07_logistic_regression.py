# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:54:52 2021

@author: kalaivanan

logistic regression 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import math
from math import sqrt 

def loReg(file):

    dataset = pd.read_csv(file)
    
    # input
    x = dataset.iloc[:, :-1].values
    # output
    y = dataset.iloc[:, -1].values
    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 1, random_state = 0)

    '''
    sc_x = StandardScaler()
    xtrain = sc_x.fit_transform(xtrain)
    xtest = sc_x.transform(xtest)
    '''

    classifier = LogisticRegression(random_state = 0)
    classifier.fit(xtrain, ytrain)

    y_pred = classifier.predict(xtest)
    cm = confusion_matrix(ytest, y_pred)
    xtest = np.array(xtest).reshape(1,len(xtest))[0]
    xtest.sort()
    ey = 1/(1+ math.e**-(classifier.intercept_[0]+classifier.coef_[0]*x ))
    print(ytest, y_pred)
    print(classifier.coef_)
    print(classifier.intercept_)
    print(np.array(xtest), y_pred)
    plt.scatter(x, ey)

def simoid_act(result):
    res = 1/(1+ math.e**(-result))
    return res

def mod_optimize(x, y_train, m, b ):
    l = x.shape[0]
    y_test = simoid_act(m*x + b)
    #print(y_test)
    #print(y_train)
    cost = 1/l*(np.sum( -y_train*np.log(y_test)-(1-y_train) * np.log(1-y_test)))
    dw = 1/l*(np.sum(x*(y_test - y_train)))
    db = 1/l*(np.sum(y_test - y_train))
    g_des = [dw, db]
    return cost, g_des

def logistic_regression(file, n):
    cost = []
    xd = pd.read_csv(file)
    xd = np.array(xd)
    m = 0
    b = 0
    l_r = .001 
    
    for i in range(n):
        (c, gdes) = mod_optimize(xd[:, 0], xd[:, -1], m, b)
        cost.append(c)
        m = m - l_r * gdes[0]
        b = b - l_r * gdes[1]

    print(m, b)
    plt.plot(range(n), cost, 'grey')
    plt.show()
    result = m * xd[:, 0] + b
    res = 1/(1+ math.e**(-result))
    plt.plot(xd[:, 0],  res, 'blue')

def likelihood(file):
    xd = pd.read_csv(file)
    xd = np.array(xd)
    for i in range(5):
        ey = []
        ey = 1/(1+ math.e**-((1.06563174*i)+(-0.11410)*xd[:,0]))
        lkh = np.concatenate((xd[:, -1].reshape(-1, 1), ey.reshape(-1,1)), axis=1)
        print(ey.shape)
        prd_1 = np.array([])
        prd_0 = np.array([])
        prd_1 = [eac[1] for eac in lkh[:, ] if eac[0] == 1]
        prd_0 = [1- eac[1] for eac in lkh[:, ] if eac[0] == 0]
        #print(prd_1 , prd_0)
        prd_1 = np.log(prd_1)
        prd_0 = np.log(prd_0)
        #print(prd_1 , prd_0)
        a = np.sum(prd_1)
        b = np.sum(prd_0)
        print(a, b, a+b)
        plt.scatter(xd[:,0], ey)

# purposely going in wrong direction 
# it did gave the sigmoid graph 
# m and b. but not matching with one got from sci library
# used rsquared as cost function. 
def meany(x, y_test):
    y = 0*x + .5
    e = 1/(1+ math.e**-(y))
    stdy = sqrt(sum((y-e)**2)/(len(x)-1))
    return stdy

def rsqrd(x, y_test, m, b, stdy):
    y_p = m*x + b
    e = 1/(1+ math.e**-(y_p))
    lse = sqrt(sum((y_test-e)**2)/(len(x)-1))
    #print(lse, std(y))
    #lse = (m*x + b)
    return ((lse/ stdy))

def gradient_des(file):
    xd = pd.read_csv(file)
    xd = np.array(xd)
    ey = np.array([])
    n = xd.shape[0] 
    l_r = .01
    m = 0
    b = 0
    x_axis = []
    y_axis = []
    z_axis = []    
    stdy = meany(xd[:, 0], xd[:, 1])
    for i in range(1000000):
        y_pred = m*xd[:, 0] + b
        ey = 1/(1+ math.e**-(y_pred))
        d_m = -2/n*sum(xd[:, 0]*(xd[:, 1]-ey))
        d_c = -2/n*sum(xd[:, 1]-ey)
        m = m - l_r*d_m
        b = b - l_r*d_c
        l_rsqrd = rsqrd(xd[:, 0], xd[:, 1], m, b, stdy)
        
        x_axis.append(m)
        y_axis.append(b)
        z_axis.append(l_rsqrd)

    print(xd[:, 0], ey)
    plt.plot(xd[:, 0], ey)
    plt.show()

    ax = plt.axes(projection='3d')
    ax.scatter(x_axis, y_axis, z_axis, c=z_axis, cmap='viridis', linewidth=1)
    plt.show()

    '''
    print(x_axis[9900:])
    print(y_axis[9900:])
    print(z_axis[9900:])
    '''

    '''
    plt.scatter(range(1000), x_axis)
    plt.show()
    plt.scatter(range(1000), y_axis)
    plt.show()
    plt.scatter(range(1000), z_axis)
    plt.show()
    '''    