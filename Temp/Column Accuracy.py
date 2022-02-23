# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:42:28 2022

@author: RPKR
"""


#Classification of cancer dignosis
#importing the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#importing the dataset 
dataset = pd.read_csv('temp.csv')
X = dataset.iloc[:, 2:31].values
Y = dataset.iloc[:, 1].values
X1= dataset.iloc[:, [2,12,22]].values
X2= dataset.iloc[:, [3,13,23]].values
X3= dataset.iloc[:, [4,14,24]].values
X4= dataset.iloc[:, [5,15,25]].values
X5= dataset.iloc[:, [6,16,26]].values
X6= dataset.iloc[:, [7,17,27]].values
X7= dataset.iloc[:, [8,18,28]].values
X8= dataset.iloc[:, [9,19,29]].values
X9= dataset.iloc[:, [10,20,30]].values
X10= dataset.iloc[:, [11,21,31]].values

dataset.head()

print("\n\nAccuracy of Individual features")
print("Cancer data set dimensions : {}".format(dataset.shape))

#Encoding categorical data values 
labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
X_tranSet, X_testSet, Y_tranSet, Y_testSet = train_test_split(X, Y, test_size = 0.25, random_state = 0)
X_tS1, X_sS1, Y_tranSet, Y_testSet = train_test_split(X1, Y, test_size = 0.25, random_state = 0)
X_tS2, X_sS2, Y_tranSet, Y_testSet = train_test_split(X2, Y, test_size = 0.25, random_state = 0)
X_tS3, X_sS3, Y_tranSet, Y_testSet = train_test_split(X3, Y, test_size = 0.25, random_state = 0)
X_tS4, X_sS4, Y_tranSet, Y_testSet = train_test_split(X4, Y, test_size = 0.25, random_state = 0)
X_tS5, X_sS5, Y_tranSet, Y_testSet = train_test_split(X5, Y, test_size = 0.25, random_state = 0)
X_tS6, X_sS6, Y_tranSet, Y_testSet = train_test_split(X6, Y, test_size = 0.25, random_state = 0)
X_tS7, X_sS7, Y_tranSet, Y_testSet = train_test_split(X7, Y, test_size = 0.25, random_state = 0)
X_tS8, X_sS8, Y_tranSet, Y_testSet = train_test_split(X8, Y, test_size = 0.25, random_state = 0)
X_tS9, X_sS9, Y_tranSet, Y_testSet = train_test_split(X9, Y, test_size = 0.25, random_state = 0)
X_tS10, X_sS10, Y_tranSet, Y_testSet = train_test_split(X10, Y, test_size = 0.25, random_state = 0)


#Feature Scaling
stdScl = StandardScaler()
X_tranSet = stdScl.fit_transform(X_tranSet)
X_testSet = stdScl.transform(X_testSet)
X_tS1 = stdScl.fit_transform(X_tS1)
X_sS1 = stdScl.transform(X_sS1)
X_tS2 = stdScl.fit_transform(X_tS2)
X_sS2 = stdScl.transform(X_sS2)
X_tS3 = stdScl.fit_transform(X_tS3)
X_sS3 = stdScl.transform(X_sS3)
X_tS4 = stdScl.fit_transform(X_tS4)
X_sS4 = stdScl.transform(X_sS4)
X_tS5 = stdScl.fit_transform(X_tS5)
X_sS5 = stdScl.transform(X_sS5)
X_tS6 = stdScl.fit_transform(X_tS6)
X_sS6 = stdScl.transform(X_sS6)
X_tS7 = stdScl.fit_transform(X_tS7)
X_sS7 = stdScl.transform(X_sS7)
X_tS8 = stdScl.fit_transform(X_tS8)
X_sS8 = stdScl.transform(X_sS8)
X_tS9 = stdScl.fit_transform(X_tS9)
X_sS9 = stdScl.transform(X_sS9)
X_tS10 = stdScl.fit_transform(X_tS10)
X_sS10 = stdScl.transform(X_sS10)
 
def per(XT,XS,modName):
  modelLoader = LogisticRegression(random_state = 0)
  modelLoader.fit(np.nan_to_num(XT), Y_tranSet)
  Y_pred = modelLoader.predict(XS)
  ap1= accuracy_score(Y_testSet, Y_pred)*100
  modelLoader = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  modelLoader.fit(np.nan_to_num(XT), Y_tranSet)
  Y_pred = modelLoader.predict(XS)
  ap2= accuracy_score(Y_testSet, Y_pred)*100
  modelLoader = SVC(kernel = 'linear', random_state = 0)
  modelLoader.fit(np.nan_to_num(XT), Y_tranSet)
  Y_pred = modelLoader.predict(XS)
  ap3= accuracy_score(Y_testSet, Y_pred)*100
  modelLoader = SVC(kernel = 'rbf', random_state = 0)
  modelLoader.fit(np.nan_to_num(XT), Y_tranSet)
  Y_pred = modelLoader.predict(XS)
  ap4= accuracy_score(Y_testSet, Y_pred)*100
  modelLoader = GaussianNB()
  modelLoader.fit(np.nan_to_num(XT), Y_tranSet)
  Y_pred = modelLoader.predict(XS)
  ap5= accuracy_score(Y_testSet, Y_pred)*100
  modelLoader = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  modelLoader.fit(np.nan_to_num(XT), Y_tranSet)
  Y_pred = modelLoader.predict(XS)
  ap6= accuracy_score(Y_testSet, Y_pred)*100
  modelLoader = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  modelLoader.fit(np.nan_to_num(XT), Y_tranSet)
  Y_pred = modelLoader.predict(XS)
  ap7= accuracy_score(Y_testSet, Y_pred)*100
  print("\n" + modName + "--> Max Accuracy : {}".format(max(ap1,ap2,ap3,ap4,ap5,ap6,ap7)))

	

per(X_tranSet, X_testSet,"Total data")
per(X_tS1,X_sS1,"Radius")
per(X_tS2,X_sS2,"texture")
per(X_tS3,X_sS3,"perimeter")
per(X_tS4,X_sS4,"area")
per(X_tS5,X_sS5,"smoothness")
per(X_tS6,X_sS6,"compactness")
per(X_tS7,X_sS7,"concavity")
per(X_tS8,X_sS8,"concave")
per(X_tS9,X_sS9,"symmetry")
per(X_tS10,X_sS10,"fractal_dimension")
