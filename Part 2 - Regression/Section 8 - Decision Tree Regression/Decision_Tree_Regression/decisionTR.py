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

# #this plot gives a curve not split lines in the data
# plt.scatter(X, y, color ="blue")
# plt.plot(X, regressor.predict(X), color="red")
# plt.title("Truth or Bluff (Decision Tree Regression)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

# higher resolution visualization plot
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color ="blue")
plt.plot(X_grid, regressor.predict(X_grid), color="red")
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()



