# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
ssX = StandardScaler()
ssY = StandardScaler()
x = ssX.fit_transform(x)
y = ssY.fit_transform(y)


# SVR
from sklearn.svm import SVR
vRegressor = SVR(kernel='rbf')
vRegressor.fit(x,y)

np.set_printoptions(precision=2)

# REVERSE PREDICTION SCALING
rr = ssY.inverse_transform(vRegressor.predict(x).reshape(-1,1))

# VISUALISATION
plt.scatter( ssX.inverse_transform(x) ,ssY.inverse_transform(y),color='red')
plt.plot(ssX.inverse_transform(x), rr , color='blue')
plt.show()