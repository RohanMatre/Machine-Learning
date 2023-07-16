# SRV 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x, y.ravel())

# Predicting a new result
new_x = sc_x.transform(np.array([[6.5]]))
y_pred = regressor.predict(new_x)
y_pred = sc_y.inverse_transform(y_pred)

# Visualizing the SVR results
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.scatter(new_x, y_pred, color='green', marker='x', label='Predicted')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.show()
