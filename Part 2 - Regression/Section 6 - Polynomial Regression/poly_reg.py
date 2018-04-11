# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# while building ML models, specially regression models, we require the features 
# to be a matrix of features. not a vector of features. Hence, X is defined as 
# X = dataset.iloc[:, 1:2].values and not X = X = dataset.iloc[:, 1]
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
print(X,y)

# very small dataset so not enough data to train and test. Hence, doesn't
# sense to split the data into train and test
# since accurate prediction is required. Entire data will be trained

# fitting linear regression model
# doing both linear and poly just for the sake of comparison
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


# fitting a polynomial linear regression model
from sklearn.preprocessing  import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
# print(X_poly)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)