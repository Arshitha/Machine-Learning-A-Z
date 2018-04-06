#dataset 
# anonymised dataset of start ups
# administrative costs = paying employees, etc

# goal: venture capitalist has hired me to create a model that 
# predicts or tells which start up is most promising and should be
# invested in

# profit is dependent variable

# multiple linear regression - more than one independent variable
# Assumptions of a linear regression:
# 1. Linearity
# 2. Homoscedasticity
# 3. Multivariate normality 
# 4. Independence of errors
# 5. Lack of multicollinearity

# before building a linear regression model you should if these
# assumptions are true. This is more of cautionary note, while 
# modelling a linear regression model in real life. It won't be dealt
# with in this course. 

# Dummy Variables for state column
# State is a categorical variable and hence, this requires pre processing
# phenomenon of dummy variable trap 
# dummyVar1 = 1 - dummyVar2, this equation implies dummyVar1 is dependent on 
# dummyVar2 and hence what happens is that the model cannot distinguish
# the effects of dummyVar1 from dummyVar2. 


# Building a model step by step	
# Question 1: Which features to include in building a model and which 
# ones to throw out? 
# 1. Garbage in, garbage out
# 2. Need to explain the correlations between the independent variables
# and the dependent variables more than just mathematically.

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4]

# encoding categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding the dummy variable trap
#X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4)
#print(X_test, y_test)
# feature scaling taken care of by the library itself

# Fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # created an object of the class linear regression
regressor.fit(X_train, y_train)

# prediction on test set
y_pred = regressor.predict(X_test)
print(y_pred)


