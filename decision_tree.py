# Decision_Tree Regression

# import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Splitting the dataset into training set and test set
''' from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)'''

# Feature Scaling
''' from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

# Fitting the Regression model to the dataset 
# Create your regressor here 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

# Predicting a new results 
x_pred = regressor.predict([[6.5]])

# Visualising the Regression results 
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color ='blue')
plt.title('Truth or Bluff (Secision Tree Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# x_grid = np.arange(min(x),max(x),0.1)
# x_grid = x_grid.reshape((len(x_grid),1))

