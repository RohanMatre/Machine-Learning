# Data Prepocessing 

# 1. Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# 2. Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,  :-1].values  # [:, :-1] - left : means take all the lines & : Right means take all columns except last one.
y = dataset.iloc[:,  3].values    # [:, 3] taking col 3 only


# 3. Splitting Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Imp Points  --> Overfeeding --> prevent by Regularity Techique 
# 1. we need to make 2 different dataset --> a. Train b. Test
# a. Train - Train set on which ML model Learns
# b. Test - Test set in which we test if ML model learns correctly corelations

# 4. Feature Scaling 
# Euclidean Distance Formula Between P1 & P2 - sqrt((x2-x1)^2 + (y2-y1)^2)
# Standardisation - 
# Xstand = x - mean(x)/standard deviation(x)
# Normalisation - 
# Xnorm = x - min(x)/max(x) - min(x)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler() 
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


'''
#  Taking Care of Missing Data
# np.set_printoptions(threshold=np.nan)
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# Learn about this Imputer library has been depreciated for google.
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN',strategy='mean',axis = 0)
# imputer = imputer.fit(x[:, 1:3])
# x[: ,1:3] = imputer.transform(x[: ,1:3])

# IMP THIS CODE!!! For missing data
# from sklearn.impute import  SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer = imputer.fit(x[:, 1:3])
# x[:, 1:3] = imputer.transform(x[:, 1:3])
'''
'''
# Encoding Categorial Data
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# labelencoder_x = LabelEncoder()
# x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# x =  onehotencoder.fit_transform(x).toarray() #

# This is Only for Country Column

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],  # The column numbers to be transformed
    remainder='passthrough'  # Leave the rest of the columns untouched
)

x = ct.fit_transform(x)
x = np.array(x, dtype=np.float64)  # convert the result to a numpy array

# This is Only for Purchased Column
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Dummy Encoding 
# OneHotEncoder

'''


# Que - Do we need to Scale the dummy Variables ??? --> ans - depends on Context(interpretation)





























