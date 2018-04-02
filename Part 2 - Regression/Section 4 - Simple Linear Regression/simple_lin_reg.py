# Simple Linear Regression
# notes from the video
# eqn for linear regression is of the form 
# y = b0 + b1x1, where b0, b1 are coefficients (or weights) x1 (independent variable)
# on which y is dependent. all of the x's are features 
# x = experience
# y = salary
# salary = b0 + b1*experience
# b0 is the y-intercept and b0 = 30k will be the base salary or salary for an entry level
# position. b1 is rate of increase in salary with per year increase in experience
# which is also the slope of the line that fits the data

# STEP 1
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting the data into test and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=4)
#print(X_train)

# Feature Scaling
# for simple linear regression models the library itself will
# take care of feature scaling and hence we don't have to do it 
# here. this is not always the case and exceptions will be mentioned
# along the course. 

# STEP 2
# sklearn library will be used
# linear regression class will be used
# object of the class is called regressor
# regressor object will be fit to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# based on the training data the regressor object will be fit
# learns the corelations between salary and experience
regressor.fit(X_train, y_train)
print("here")

# STEP 3
# predictions on the test set based on training in the prev set
y_pred = regressor.predict(X_test)
print(y_test, y_pred)

# STEP 4: 
# visualizing training set results
plt.scatter(X_train, y_train, color= "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test, y_test, color= "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

