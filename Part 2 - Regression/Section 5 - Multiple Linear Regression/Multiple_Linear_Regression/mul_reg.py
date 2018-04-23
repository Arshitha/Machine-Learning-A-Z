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
X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=345)
# print(X_test, y_test)
# feature scaling taken care of by the library itself

# Fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # created an object of the class linear regression
regressor.fit(X_train, y_train)

# prediction on test set
y_pred = regressor.predict(X_test)
#print(y_pred)

# some of the independent variables in the dataset could have higher statistical 
# significance than the other independent variables on the dependent variable.
# If non-statistically significant variables are removed, then we'd have a 
# team of independent variables that are highly effective in predicting output 
# labels

# backward elimination preparation
import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((50,1)).astype(int), values=X,axis=1)
#print(X)

X_opt = X[:, [0,1,2,3,4,5]] 
# since for back elimination we are using a different lib we won't be using 
# the regressor model. Instead, we'd be using a new regressor ( basically, multiple reg model)
# but it'll be using the statsmodels lib ratger than linear regression lib
# ols = ordinary least squares
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
# the summary function 

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())





