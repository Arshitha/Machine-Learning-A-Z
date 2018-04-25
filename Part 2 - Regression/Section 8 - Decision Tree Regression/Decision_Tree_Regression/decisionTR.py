# Decision Tree Regression

# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fitting the decision tree regression model to our dataset 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

# predicting a new result
y_pred = regressor.predict(6.5)
print(y_pred)

# Visualising the Decision Tree Regression results
plt.scatter(X, y, color ="blue")
plt.plot(X, regressor.predict(X), color="red")
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


