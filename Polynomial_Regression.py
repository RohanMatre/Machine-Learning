# Polynomial Regression

# import the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values  # [:, 1] -> This include vector so do it like this for matrix -> [:, 1:2]
y = dataset.iloc[:, 2].values


# Splitting the dataset into training set and testing set
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)
# Here we don't require this bcz 2 reasons - a) small dataset b) Make Accurate decision- require max. data

# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree=2)
# poly_reg = PolynomialFeatures(degree=3)
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y) 

# Visualising the Linear Regression results 
plt.scatter(x, y,color='red')
# prediction
plt.plot(x,lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results 
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y,color='red')
# prediction
# plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()