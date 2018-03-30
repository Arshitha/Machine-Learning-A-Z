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

#print(dataset)

# dealing with missing data
# replace the missing data with mean of that data
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

# fitting imputer (with small i) object to our feature matrix X
imputer = imputer.fit(X[:,1:3]) # equality because the same imputer object was fit to X
X[:,1:3] = imputer.transform(X[:,1:3])

# missing values replaced by their mean
#print(X)

# encode categorical data
# in our dataset we have 2 categorical data categories. Country and purchased
# since categories are strings and algorithms understand on;y numbers these data points
# need to be encoded as numbers

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# object of LabelEncoder class is labelencoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
# print(X)
# at this point countries have been encoded with 0,1 and 2. However, even this 
# is a problem because the algorithm could still interpret it as something
# with relational order even though that's not the case. 

# to deal with this unintended assignment of order, dummy variables are 
# created

# we'll create 3 separate columns, one for each country instead of the one comprehensive 
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
print(X)






