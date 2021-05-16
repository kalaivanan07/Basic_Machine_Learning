# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:15:24 2021

@author: kalaivanan

~calculate knearest through scikit method and my own implementation 
~distance method is used from utilities package

"""
import numpy as np
from utilities import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
 
import time 

# method 1  finding eculidean distance to each point
# sorting the distnance and findind the nearest neighbour

def knearest(xf, yf, Nknn):
    print('inside knearest..') 
    # format should be train and test file
    # same no of cols, with last col as ypred
    
    xdf = pd.read_csv("E:\\Tech\\ML\\Data_Set\\" + xf)
    ydf = pd.read_csv("E:\\Tech\\ML\\Data_Set\\" + yf)
    pred_Copy = {eac:0 for eac in xdf[xdf.columns[-1:][0]].unique()}
    #print(pred_Copy )
    global error_rate_own
    error_rate_own = []

    for knn in range(1,Nknn+1):
        st = time.time()
        build_pred_list = []
        for eac_y in ydf.itertuples():
            #print(eac_y)
            distance = []
            a = []
            b = []
            c = []
            p = []
            q = []
            r = []
            point_y  = []
            # eac_y[0]  is index. 
            point_y = [np.array(eac_y[1:-1])] * len(xdf)
            
            a = euclidean(np.array(xdf[xdf.columns[0:-1]]), point_y)
            a = a.reshape(len(a), 1)
            b = [np.array(eac_y[:])] * len(xdf) 
            c = np.array(xdf[xdf.columns[-1:]])
            
            '''
            print(a)
            print(b)
            print(c)
            
            distance format - col 1           -> euclidean dis 
                              col 2 - col n-1 -> y variables
                              col n           -> train pred 
            '''                            
            
            distance = np.concatenate((a,b,c), axis=1)
            distance = distance[distance[:, 0].argsort()]
            #print(distance)
            pred = dict(pred_Copy)
            #print(pred)
            for i in range(knn):
                for eac in pred.keys():
                    if eac == distance[i][-1]:
                        pred[eac] +=1
            #print(pred)
            p = np.array(eac_y[:])
            q = np.array([max(pred, key=pred.get)])
            r = np.concatenate((p,q), axis=0)
            #print(r)
            build_pred_list.append(r)

        build_pred_list = np.array(build_pred_list)
        print(time.time() - st )
        
        '''
        print(build_pred_list)
        print(list(build_pred_list[:, -2]))
        print(list(build_pred_list[:, -1]))
        '''

        #confusion_matrixz(list(build_pred_list[:, -2]), list(build_pred_list[:, -1]))
        #print(confusion_matrix(list(build_pred_list[:, -2]), list(build_pred_list[:, -1])))
        err = (1-accuracy_score(list(build_pred_list[:, -2]), list(build_pred_list[:, -1])))*100
        error_rate_own.append(err)
        print(time.time() - st )
        
# method 2 : using scikit library

def sci_knearest(xf, Nknn):
    print('inside sci_knearest..') 
    
    xdf = pd.read_csv("E:\\Tech\\ML\\Data_Set\\" + xf)
    X_train, X_test, y_train, y_test = train_test_split(xdf[xdf.columns[0:-1]], xdf[xdf.columns[-1]], test_size=0.2, random_state=0)
    
    '''
    print(xdf[xdf.columns[0:-1]])
    print(xdf[xdf.columns[-1]])
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)
    '''
    global error_rate_sci
    error_rate_sci = []
    for n in range(1, Nknn+1):
        st = time.time()
        knn = KNeighborsClassifier(n_neighbors = n)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        #print(confusion_matrix(y_test, y_pred))
        #print(classification_report(y_test, y_pred))
        err = (1 - accuracy_score(y_test, y_pred))*100
        error_rate_sci.append(err)
        print(st - time.time())
        
def plot_knn_gradient():
    plt.figure(figsize =(10, 6))
    plt.plot(range(1, len(error_rate_own)+1),
             error_rate_own, color ='blue', 
             linestyle ='dashed', 
             marker ='o', 
             markerfacecolor ='red', 
             markersize = 10)
    plt.plot(range(1, len(error_rate_sci)+1),
             error_rate_sci, color ='grey', 
             linestyle ='dashed', 
             marker ='o', 
             markerfacecolor ='red', 
             markersize = 10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')

def main(xf, yf, Nknn):
    knearest(xf, yf, Nknn)
    sci_knearest(xf, Nknn)
    plot_knn_gradient()
