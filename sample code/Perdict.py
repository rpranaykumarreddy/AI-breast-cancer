# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:42:28 2022

@author: RPKR
"""


#Classification of cancer dignosis
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset 
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 2:32].values
Y = dataset.iloc[:, 1].values

dataset.head()

dataframe = pd.DataFrame(Y)

#Encoding categorical data values 
from sklearn.preprocessing import LabelEncoder
labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

inp = [[7.76, 24.54, 47.92, 181, 0.05263, 0.04362, 0, 0, 0.1587, 0.05884, 0.3857, 1.428, 2.548, 19.15, 0.007189, 0.00466, 0, 0, 0.02676, 0.002783, 9.456, 30.37, 59.16, 268.6, 0.08996, 0.06444, 0, 0, 0.2871, 0.07039]]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
stdScl = StandardScaler()
X = stdScl.fit_transform(X)
inp = stdScl.transform(inp)
 
#Fitting Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
modelLoader = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
modelLoader.fit(X, Y)

output= modelLoader.predict(inp)
print(output)