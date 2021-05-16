# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:06:27 2021

@author: kalaivanan
Decision tree algorithm 

entropy - scikit
Gini    - scikit 
plot the ouput
efficiency - confusion matrix 

"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np 
  
# Importing the required packages
  
# Function importing Dataset
def impData(file):
    df = pd.read_csv('E:\\Tech\\ML\\Data_Set\\'+file,  sep= ',', header = 1)
    print ("Dataset Length:  ", len(df))
    print ("Dataset Shape: ", df.shape)
    print ("Dataset: ",df.head())
    return df

#  Function to split the dataset
def splitdataset(df):

    # Separating the target variable
    X = []
    Y = []
    X = df.values[:, 0:-2]
    Y = df.values[:, -1]

    # Splitting the dataset into train and test
    X_train = [] 
    X_test  = [] 
    y_train = []
    y_test  = []
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state =  np.random)

    #print(len(X_train))
    #print(len(X_test))
    #print(len(y_train))
    #print(len(y_test.count())

    return (X, Y, X_train, X_test, y_train, y_test)

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = np.random,max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini
      
# Function to perform training with entropy.
def train_using_entropy(X_train, X_test, y_train):
  
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = np.random,
            max_depth = 3, min_samples_leaf = 5)
  
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
  
# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred  = []
    y_pred = clf_object.predict(X_test)
    #print("Predicted values:")
    #print(y_pred)
    return y_pred
      
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
      
    #print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    #print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
    #print("Report : ", classification_report(y_test, y_pred))
    err = 100 - accuracy_score(y_test,y_pred)*100
    return err
    
# Driver code
def main(nTrain, file):

    global err_entropy
    global err_gini
    err_entropy = []
    err_gini = []
    # Building Phase
    data = impData(file)
    
    for i in range(nTrain):
        X = []
        Y = []
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
        clf_gini = train_using_gini(X_train, X_test, y_train)

        X = []
        Y = []
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
        clf_entropy = train_using_entropy(X_train, X_test, y_train)

        # Operational Phase
        #print("Results Using Gini Index:")
          
        # Prediction using gini
        y_pred_gini = []
        y_pred_gini = prediction(X_test, clf_gini)
        err_gini.append(cal_accuracy(y_test, y_pred_gini))
        
        #print("Results Using Entropy:")
        # Prediction using entropy
        y_pred_entropy = []
        y_pred_entropy = prediction(X_test, clf_entropy)
        err_entropy.append(cal_accuracy(y_test, y_pred_gini))
       
        
    #print(err_entropy)
    #print(err_gini)
    plot_grad_descent(nTrain)

def plot_grad_descent(nTrain):
    plt.figure(figsize =(10, 6))
    
    plt.plot(range(nTrain), 
             err_gini, 
             color = 'blue', 
             linestyle='dashed', 
             marker='o', 
             markerfacecolor='red', 
             markersize = 10)
    plt.plot(range(nTrain), 
             err_entropy,
             color='grey',
             linestyle= 'dashed',
             marker='o',
             markerfacecolor= 'red',
             markersize = 10)

    plt.xlabel('No. of splits')
    plt.ylabel('Error Rate') 
    plt.title('Error Plot')
    
    
'''     
# Calling main function
if __name__=="__main__":
    main()    
'''
