# Data Preprocessing

# 3 essential libraries
import numpy as np 
import matplotlib.pyplot as plt #pyplot is a sub library of matplotlib
import pandas as pd # one of the best libraries to import and manage large data sets

# Importing the libraries
from sklearn.preprocessing import Imputer # library that conatains a lot ML algorithms
# it also contains preprocessing library that contains classes, methods etc for
# pre processing of datasets
# Imputer class takes care missing data

# importing the dataset 
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:, :-1].values # matrix of features
y = dataset.iloc[:,3].values # vector of labels

# print(dataset)

# dealing with missing data
# replace the missing data with mean of that data
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

# fitting imputer (with small i) object to our feature matrix X
imputer = imputer.fit(X[:,1:3]) # equality because the same imputer object was fit to X
X[:,1:3] = imputer.transform(X[:,1:3])

# missing values replaced by their mean
print(X)


