# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:47:46 2020

"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



"""
Below is an example for numeric features 
"""

### Data path
csv_path = "EnergyEfficiency_data.csv"


### Define feature/label
data_info = pd.read_csv(csv_path, header=0) #header=0 -> no column names

x_arr = np.asarray(data_info.iloc[:, 0:8])
y_arr = np.asarray(data_info.iloc[:, 8:10])


min_max_scaler = preprocessing.MinMaxScaler() ### to 0 ~ 1
max_abs_scaler = preprocessing.MaxAbsScaler() ### to -1 ~ 1

x_arr = max_abs_scaler.fit_transform(x_arr)

#print(data_info.columns)

### Split into train/test set
X_train, X_test, Y_train, Y_test = train_test_split(x_arr, y_arr, test_size=0.2, random_state=0)

### Save/Load npz file
np.savez('preprocessed_data.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
Data = np.load('preprocessed_data.npz')
X_train = Data["X_train"]
X_test = Data["X_test"]
Y_train = Data["Y_train"]
Y_test = Data["Y_test"]


"""
Below is an example for non-numeric features transform into numeric features
"""


### Create a demonstrate panda frame with non-numeric features
df2 = pd.DataFrame(
        [["Green", "M", 10.1, 1],
         ["Red", "L", 9.4, 2], 
         ["Blue", "XL", 13.1, 1]]      
        )
df2.columns = ["Color", "Size", "Price", "Label"]




### Create a mapping dict
size_mapping = {
        "XL":3,
        "L":2,
        "M":1    
        }

df2["Size"] = df2["Size"].map(size_mapping) # original dataframe will be overwritten 



### One-hot encoding
one_hot_encoding = pd.get_dummies(df2["Color"], prefix="Color") # original dataframe will not be overwritten 
df2 = df2.drop(["Color"], axis=1)

### After feature transformation
df2 = pd.concat([one_hot_encoding, df2], axis=1) # concatate dataframes
x_arr = np.asarray(df2.iloc[:, 0:-1])
y_arr = np.asarray(df2.iloc[:, -1:])
#X_train, X_test, Y_train, Y_test = train_test_split(x_arr, y_arr, test_size=0.2, random_state=0)





























