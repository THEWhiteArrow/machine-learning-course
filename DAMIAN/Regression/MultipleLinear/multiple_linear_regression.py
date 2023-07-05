# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Missing data

# avg approach
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=0, strategy='mean')
# x[:,:-1] = imputer.fit_transform(x[:,:-1])

# remove approach


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [3])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.2,random_state=1)

print(xTrain)
# Training the Multiple Linear Regression model on the Training set 
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(xTrain,yTrain)



# Predicting the Test set results 
yPred = regression.predict(xTest)
np.set_printoptions(precision=2)

# print(yTest)
# print(yPred)


comparison = np.concatenate(
        (
            yPred.reshape(len(yPred),1),
            yTest.reshape(len(yTest),1),
        ),
        1
    )
diff = 0

# print(comparison)

for row in comparison:
    diff += row[1]-row[0]

print(diff)

