# Multiple Linear Regression

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv('50_Startups.csv')

# Set the x(independent) and y(dependent) data's
x = dataset.iloc[:, :-1].values   # this is x so -1 atlast column remove
y = dataset.iloc[:, 4].values     # this is y so only 4 column which dependent variable is select 


# Firstly, we require the x dataset to convert the word text into categorial data.
# Encoding Categorial Data
# This is Only for State Column

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],  # The column numbers to be transformed
    remainder='passthrough'  # Leave the rest of the columns untouched
)
x = ct.fit_transform(x)
x = np.array(x, dtype=np.float64)  # convert the result to a numpy array

# Avoiding the dummy variable trap
x = x[:, 1:]  # this is remove one first dummy variable


# Splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predicting the test set results
y_pred = regressor.predict(x_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
# Assuming `x` and `y` are defined appropriately
# Add a column of ones to the beginning of `x`
# Step-1
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)
# Step-2
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# Step-3
regressor_OLS.summary()

# Here x3 has the highest p-value>5% so x3 has remove predictor
x_opt = x[:, [0, 1, 2, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# Step-3
regressor_OLS.summary()


x_opt = x[:, [0, 1, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# Step-3
regressor_OLS.summary()

x_opt = x[:, [0, 1, 4]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# Step-3
regressor_OLS.summary()








