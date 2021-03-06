# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:06:27 2021
@author: kalaivanan

-----------------------------------------------------------
	Linear regression:
-----------------------------------------------------------

Conclusion: 
-----------
Except for covariance method all other method's output are similar.  
std is calculated for "n" and variance is calculated for "n-1"
And hence the difference in op of covariance

Partial derivation of least squarred :
--------------------------------------
The derivation is available in Khan Academy -> Stats & Probability -> Linear Regression 
The formula for finding optimal "m" and "b" is given. 
Given "m", "b", and the testing data, y can be predicted.

Correlation Coefficient:
------------------------
m = coefficient correlation *  (std(y)/ std(x))
any best line passes through mean x and mean y. find b using this

Covariance Method:
-----------------
m = covariance(x,x)/ (std(x) * std(x))
any best line passes through mean x and mean y. find b using this

Gradient Descent:
-----------------
1. Take derivative of loss function, w.r.t m and w.r.t b.
2. find y prediction using above m and b (first iteration is assumption)
3. find d_m and d_b using the derivative equations
4. using learning rate and the above values, calculate new m and b
5. repeate 2 to 4 until loss is minimum 

Python Library:
---------------
Using Scikit Linear regression

Efficiency of line:
------------------
r squared calculated in each method

3D Analysis:
------------
Plot to analyse the gradient descent

Excel:
------
Step 1: write down values of X and Y (training data)
Step 2: Select the data, and from menu, choose Insert -> Scatter
Step 3: Choose Design - > chart layouts, choose layout 9,  for linear regression.
Step 4 : The line can be used to predict the testing data.

pending:
---------

real time project uning multivariate linear regression using matrix :

"""

from math import sqrt
from numpy import multiply 
from numpy import mean
from numpy import std
from numpy import cov
from numpy import array 
from scipy.stats import pearsonr
import matplotlib.pyplot as plt  # To visualize
from sklearn.linear_model import LinearRegression
import pandas as pd  # To read data
import time

x =  array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
y =  array([ 4.5 ,  5.6 ,  7.6 ,  5.8 ,  7.  ,  8.75,  8.  ,  8.4 , 11.8 , 7.6 , 10. ])
# blr - bivariate linear regression 

def rsqrd(x,y, m, b):
    lse = sqrt(sum((y-(m*x + b))**2)/(len(x)-1))
    #print(lse, std(y))
    #lse = (m*x + b)
    return ((lse/ std(y)))


def blr_drv(x, y):
    # based on mathematical derivation - khan academy stats & prob
    # m = (mean(x.y) - (mean(x).mean(y)))/ (mean(x^2)- (mean(x))^2)
    # b = mean(y) - m * mean(x)
    
    mx = mean(x)
    my = mean(y)
    mxsqr = mean(multiply(x, x))
    mxy = mean(multiply(x,y))
    m = ((mx * my) - (mxy)) / (mx**2 - mxsqr)
    b = my - m * mx
    l_rsqrd = rsqrd(x,y, m, b)

    print('blr_drv--> slope= %.3f y-intercept= % .3f  rsqrd= %.3f ' % (m, b, l_rsqrd))
    return((m*(x) + b))

def blr_crr_cff(x,y):
    p_cor = pearsonr(x,y)
    m = p_cor[0]*(std(y)/std(x))
    print(p_cor, std(y), std(x))
    b = mean(y) - m*mean(x)     # best fit passes through the mean(x), mean(y)
    l_rsqrd = rsqrd(x,y, m, b)
    
    print('blr_crr_cff--> slope= %.3f y-intercept= % .3f rsqrd= % .3f ' % (m, b, l_rsqrd))
    return((m*(x) + b))
    
def blr_cov_var(x,y):
    m = cov(x,y)[0,1] / std(x)**2
    # m = (sum((x - mean(x))*(y - mean(y)))/std(x)**2)/(len(x)-1)
    print(m, std(x))
    b = mean(y) - m*mean(x)
    l_rsqrd = rsqrd(x,y, m, b)
    
    print('blr_cov_var--> slope= %.3f y-intercept= % .3f rsqrd= % .3f ' % (m, b, l_rsqrd))
    return((m*(x) + b))
    
def blr_sci(x, y):
    
    '''
    data = pd.read_csv('data.csv')  # load data set
    X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    '''
    
    x  =  array(x).reshape(-1, 1)
    y  =  array(y).reshape(-1, 1)
    
    l_r = LinearRegression()  # create object for the class
    l_r.fit(x, y)  # perform linear regression
    y_pred = l_r.predict(x)  # make predictions
   
    print('blr_sci--> slope= %.3f y-intercept= % .3f rsqrd= % .3f ' % (l_r.coef_, l_r.intercept_, l_r.score(x,y)))
    return(y_pred)

def gradient_descent(x,y):
    
    m = 0
    b = 0 
    l_r = .01 
    n = len(x) 
    
    for i in range(5000):
        y_pred = m*x + b 
        d_m = -2/n*sum(x*(y-y_pred))
        d_c = -2/n*sum(y-y_pred)
        m = m - l_r*d_m
        b = b - l_r*d_c

    l_rsqrd = rsqrd(x,y, m, b)
    plt.scatter(x, (m*x +b))
    plt.plot([x[0], x[len(x)-1]], [(m*x[0] +b), (m*x[len(x)-1] +b)])
    print('gradient_descent--> slope=%.3f y-intercept=%.3f lsqrd=%.3f' % (m, b, l_rsqrd))
    return(m*x + b)

def main(file):

    xd = pd.read_csv(file)
    
    x = xd.iloc[:, 0]
    y = xd.iloc[:, 1]

    print(x)
    print(y)
    
    plt.scatter(x, y)
    plt.plot(x, blr_drv(x, y), color='red')
    plt.plot(x, blr_crr_cff(x, y), color='black')
    plt.plot(x, blr_cov_var(x,y), color='blue')
    plt.plot(x, blr_sci(x,y), color='green')
    plt.plot(x, gradient_descent(x,y), color = 'yellow')
    plt.show()
    gd_3d(x,y)
    # from this graph except for covariance graph all lines coincides
    # in python std is calculated for "n" and variance is calculated for "n-1"
    # using numpy and hence the difference 
    
def gd_3d(x,y):
    
    # 3D analysis of gradient descent of bivariate data 
    
    m = 0
    b = 0 
    l_r = .01 
    n = len(x) 
    
    x_axis = []
    y_axis = []
    z_axis = []
    
    for i in range(3000):
        y_pred = m*x + b 
        d_m = -2/n*sum(x*(y-y_pred))
        d_c = -2/n*sum(y-y_pred)
        m = m - l_r*d_m
        b = b - l_r*d_c
        l_rsqrd = rsqrd(x,y, m, b)
        x_axis.append(m)
        y_axis.append(b)
        z_axis.append(l_rsqrd)

    print(x_axis)
    print(y_axis)
    print(z_axis)
    
    ax = plt.axes(projection='3d')
    ax.scatter(x_axis, y_axis, z_axis, c=z_axis, cmap='viridis', linewidth=1)

# Calling main function
'''
if __name__=="__main__":
    main(s,y)    
'''

def gd_3d(x,y):
    
    # 3D analysis of gradient descent of bivariate data
    
    m = 0
    b = 0 
    l_r = .01 
    n = len(x) 
    
    x_axis = []
    y_axis = []
    z_axis = []
    
    oneSidem = []
    oneSideg = [] 
    previsousm = 99999
    
    for i in range(20):
        #time.sleep(1)
        '''
        if i <= 30:
            for j in range(50000000):
                None
        '''
        y_pred = m*x + b 
        d_m = -2/n*sum(x*(y-y_pred))
        d_c = -2/n*sum(y-y_pred)
        #print(d_m, d_c)
        m = m - l_r*d_m
        b = b - l_r*d_c
        l_rsqrd = rsqrd(x,y, m, b)
        
        x_axis.append(m)
        y_axis.append(b)
        z_axis.append(l_rsqrd)

        # one side descent
        if m < previsousm:
            previsousm = m
            oneSideg.append(l_rsqrd)
            oneSidem.append(m)
        elif m >= previsousm:
            previsousm = m
        
    print(oneSidem)
    print(oneSideg)
    oneSidem.sort()
    plt.plot(oneSidem, oneSideg)
    # one side descent         
    
    plt.plot(x_axis, z_axis)
    #plt.plot(oneSidem, oneSideg)
    
    '''
        x_axis.append(m)
        y_axis.append(b)
        z_axis.append(l_rsqrd)
    '''    
    ax = plt.axes(projection='3d')
    ax.scatter(x_axis, y_axis, z_axis, c=z_axis, cmap='viridis', linewidth=1)
    print('x-' + str(m) + 'y-' + str(b) + 'z-' + str(l_rsqrd) + '-i' + str(i))
    plt.show()
                    
    '''
        plt.xlabel('X')
        plt.ylabel('Y')
        print(x_axis)
        print(y_axis)
        print(z_axis)
        
        ax = plt.axes(projection='3d')
        ax.scatter(x_axis, y_axis, z_axis, c=z_axis, cmap='viridis', linewidth=1)
    '''
            
    '''        
    plt.scatter(range(1000), x_axis)
    plt.show()
    plt.scatter(range(1000), y_axis)
    plt.show()
    plt.scatter(range(1000), z_axis)
    plt.show()
    '''
