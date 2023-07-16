# Random Forest Regression(Non-Contiguous Regression Model)(Version of Ensemble Learning)

# import the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values 

# Fitting a Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0) # 1. Parameter - n_estimators = no. of trees in forest. 2. Criterion = default. 3. random_state 
regressor.fit(x,y)

# predicting a new results s
y_pred = regressor.predict([[6.5]]) 

# Visualising the Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('Truth or Bluff(Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# No. of Predict with 1 tree - 150000
# No. of Predict with 10 tree - 167000
# No. of Predict with 100 tree - 158300 
# No. of Predict with 300 tree - 160333.333.. 
