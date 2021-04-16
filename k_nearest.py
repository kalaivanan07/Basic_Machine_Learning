# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:15:24 2021

@author: kalaivanan

~calculate knearest through scikit method and own implementation 
~distance method is used from utilities package

"""

from utilities import * 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

'''
test data 
x  = {0:[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7)],
          1:[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]}
'''

# method 1  finding eculidean distance to each point
# sorting the distnance and findind the nearest neighbour

def knearest(x, y):

    distance=[]

    for eac in x:
        distance = list(zip(euclidean(x[eac], y), [eac] * len(x[eac]))) + distance 
        #print(distance) for debugging
        #print(x[eac])

    distance.sort()
    # print(distance)
    pred = { eac:0 for eac in x}

    for i in range(3):
        for eac in x:
            if eac == distance[i][1]:
                pred[eac] +=1

    print(pred)
    
# method 2 : using scikit library 

def sci_knearest(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(pred)
    # evaluating our KNN model ! 

    print(confusion_matrix(y_train, pred))
    print(classification_report(y_test, pred))
