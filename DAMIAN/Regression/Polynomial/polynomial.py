from random import randint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values
 
# LINEAR REGRESSION

from sklearn.model_selection import train_test_split
# xTrain,xTest,yTrain,yTest = train_test_split(x,y, test_size=0.2, random_state=randint(0,4294967295))
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)


# POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
pFeatures = PolynomialFeatures(degree=4)
xPolyTrain = pFeatures.fit_transform(x)
lr2 = LinearRegression()
lr2.fit(xPolyTrain,y)



# VISUALISATION LINEAR
plt.scatter(x,y, color='red')
plt.plot(x, lr.predict(x), color='blue')
plt.title('polynomial (train)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


# VISUALISATION POLYNOMIAL
plt.scatter(x,y, color='red')
plt.plot(x, lr2.predict(pFeatures.fit_transform(x)), color='blue')
plt.title('polynomial (train)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

# VISUALISATION RE-POLYNOMIAL
xGrid = np.arange(min(x),max(x),0.1)
xGrid = xGrid.reshape(len(xGrid),1)
plt.scatter(x,y, color='red')
plt.plot(xGrid, lr2.predict(pFeatures.fit_transform(xGrid)), color='blue')
plt.title('polynomial (train)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()