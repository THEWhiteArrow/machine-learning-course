# Data Preprocessing Template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(x)
# print(y)


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(x[:,1:])
x[:,1:] = imputer.transform(x[:,1:])

# print(x)

# Encoding categorical data

# independent data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

# print(x)


# dependent data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# print(y)

# SPLITTING DATA INTO TRAINING AND TEST SETS

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.2)

# print('TRAIN')
# print(xTrain)
# print(yTrain)

# print('TEST')
# print(xTest)
# print(yTest)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

xTrain[:,3:] = ss.fit_transform(xTrain[:,3:])
xTest[:,3:] = ss.transform(xTest[:,3:])

print('TRAIN')
print(xTrain)
print(yTrain)



print('TEST')
print(xTest)
print(yTest)