import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utilities import *
from sklearn.model_selection import train_test_split
import time

def ranForest2D(file):
    st = time.time()
    #data = pd.read_csv('E:\\Tech\\ML\\Data_Set\\01_balance_scale\\01_balance_scale_train_rf.csv')
    data = pd.read_csv(file)
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :].values, data.iloc[:, -1].values, test_size=0.2, random_state=0)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test)
    print(y_test)
    rs = []
    for i in range(2000, 4000, 25):
        regressor = RandomForestRegressor(n_estimators = i, random_state = 0)
        regressor.fit(x_train[:, :-1], y_train)
        y_pred = regressor.predict(x_test[:, :-1])
        del(regressor)
        rs.append(np.sum((y_pred/y_test), axis = 0)) 
        print(i)

    print(rs)
    print(y_pred)
    print(time.time() - st )    
    plt.figure(figsize =(10,8))
    plt.plot(range(2000, 4000, 25), rs, color = 'grey', linestyle ='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title('Random Forest Regression')
    plt.xlabel('Forest Trees Count')
    plt.ylabel('Percentage Error')
    plt.show()

def ranForest3D(file):
    st = time.time()
    #data = pd.read_csv('E:\\Tech\\ML\\Data_Set\\01_balance_scale\\01_balance_scale_train_rf.csv')
    data = pd.read_csv(file)
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :].values, data.iloc[:, -1].values, test_size=0.2, random_state=0)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test)
    print(y_test)
    rs = []
    for i in range(2025, 4525, 20):
        regressor = RandomForestRegressor(n_estimators = i, random_state = 0)
        regressor.fit(x_train[:, :-1], y_train)
        y_pred = regressor.predict(x_test[:, :-1])
        del(regressor)
        rs.append((y_pred/y_test))
        #rs.append(list(range(125))) 
        print(i)
    y = np.array([list(range(len(x_test)))]*len(x_test))
    x = np.array([list(range(2025, 4525, 20))]*len(x_test)).T
    rs = np.array(rs)
    print(time.time() - st)
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_wireframe(x, y, rs, color = 'grey')
