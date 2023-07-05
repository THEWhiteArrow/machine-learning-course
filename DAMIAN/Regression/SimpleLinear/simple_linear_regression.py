# Simple Linear Regression
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# split dataset into training and test sets
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest =  train_test_split(x,y,test_size=0.2 , random_state=1)

# LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(xTrain, yTrain)  

yPred = regressor.predict(xTest)


# VISUALISATION

# training set
plt.scatter(xTrain,yTrain,color='red')
plt.plot(xTrain, regressor.predict(xTrain), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# test set
plt.scatter(xTest,yTest,color='red')
plt.plot(xTrain, regressor.predict(xTrain), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()