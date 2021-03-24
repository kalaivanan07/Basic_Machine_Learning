----------------------------------------------------------
	Linear regression:
----------------------------------------------------------

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
-----------
Plot to analyse the gradient descent

Excel:
------
Step 1: write down values of X and Y (training data)
Step 2: Select the data, and from menu, choose Insert -> Scatter 
Step 3: Choose Design - > chart layouts, choose layout 9,  for linear regression. 
Step 4 : The line can be used to predict the testing data. 

The related files are : 
Test_data.xlsx 
linear_regression.py
