import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# TRAIN DECISION TREE
from  sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x,y)

xGrid = np.arange(min(x),max(x),0.1)
xGrid = xGrid.reshape(len(xGrid),1)
plt.scatter(x,y, color='red')
plt.plot(xGrid,regressor.predict(xGrid), color='blue')
plt.title('decision tree regression')
plt.show()