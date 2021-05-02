# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:15:52 2021
@author: kalaivanan
Bayes algorithm - boundary between two distribution 
probability     - conditional probability 
sci             - 
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from utilities import * 
import matplotlib.pyplot as plt 

def plot_two_pdf(x1_axis, x2_axis):
    x1_axis.sort()
    x2_axis.sort()
    plt.plot(x1_axis, my_pdf(x1_axis)[0], 'grey')
    plt.plot(x2_axis, my_pdf(x2_axis)[0], 'blue')
    return 'Hi'

def my_pdf(x_axis):
    # plotting a normal distribution using math import numpy.lib.recfunctions as rfnformula
    y_ndis = []
    mu   = mean(x_axis)
    si   = std(x_axis)
    for eac in x_axis:
        z = (eac-mu)/si
        y_ndis.append(1/(math.sqrt( 2 * math.pi * si**2 * math.e**(z**2))))
    return (y_ndis, mu, si)

def nbayes(x1_axis, x2_axis):
    # the above is simplified to below
    # simplifed pdf formula to equate to two variables
    si1 = std(x1_axis)
    si2 = std(x2_axis)
    mu1 = mean(x1_axis)
    mu2 = mean(x2_axis)
    x1_axis.sort()
    x2_axis.sort()
    boundary = []
    for i in range(len(x1_axis)):
        z1 = (x1_axis[i] - mu1)/si1
        for j in range(len(x2_axis)):
            z2 = (x2_axis[j] - mu2)/si2
            if round((si1**2 * math.e**(z1**2)),3) == round((si2**2 * math.e**(z2**2)),3):
                boundary.append((x1_axis[i], x2_axis[j]))

    plot_two_pdf(x1_axis, x2_axis)
    for eac in boundary:
        plt.axvline(x=eac[0], color='red', linestyle='--')
        plt.axvline(x=eac[1], color='blue', linestyle=':')

    # -------------
    # boundary line
    # -------------
    plt.axvline(x= mean(boundary[0]) , color='yellow', linestyle='solid')

def bayes(file, itr):
    xdf = pd.read_csv(file)
    l_uniq = [[]]
    l_uniq = [xdf[eac].unique() for eac in xdf.iloc[:, range(len(xdf.columns))]]
    d_clas = {i:0 for i in l_uniq[-1]} #child
    clas_count = { i: xdf[xdf.iloc[:, -1] == i].shape[0]  for i in d_clas.keys() }
    d_clas = {i:{j: {k:0 for k in l_uniq[j]} for j in range(len(l_uniq)-1)} for i in d_clas.keys()}
    #print(d_clas)
    #print(clas_count)
    err_rate = []
    for n in range(itr):
        x_train, x_test, y_train, y_test = train_test_split(xdf[xdf.columns[:]], xdf[xdf.columns[-1]], test_size=.2, random_state=n)
        #print(x_train)
        d_clas = {i:{j:{k:x_train[x_train.iloc[:, -1] == i][x_train.iloc[:, j] == k].shape[0] for k in l_uniq[j]} for j in range(len(l_uniq)-1)} for i in d_clas.keys()}
        #print(d_clas)
        y_pred = []
        prob_eac_class = {}
        col_length = len(l_uniq)-1

        for i in range(len(x_test)):
            prob_eac_class = {}
            for k in d_clas.keys():
                prob_eac_class[k] = 1
                for j in range(col_length):
                    prob_eac_class[k] = prob_eac_class[k] * d_clas[k][j][x_test.iloc[i][j]] / clas_count[k]

            #print(prob_eac_class)
            y_pred.append(max(prob_eac_class, key=prob_eac_class.get))

        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        #print(y_pred)
        #print(y_test)
        (metrixs, acc) = confusion_matrixz(y_test, y_pred)
        #print(metrixs)
        #print(acc)
        err_rate.append((100 - acc))
    plot(itr, err_rate)

def plot(itr, err_rate):
    plt.figure(figsize=(10,8))
    plt.plot(list(range(itr)), err_rate, linestyle='dashed', color='grey', marker='o', markerfacecolor='red', markersize=10)
    plt.title('N Bayes Classifier')
    plt.xlabel('Iteration')
    plt.ylabel('Error rate')
