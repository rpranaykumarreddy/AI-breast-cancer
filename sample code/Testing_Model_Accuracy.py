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
X = dataset.iloc[:, 2:13].values
Y = dataset.iloc[:, 1].values

dataset.head()

print("Cancer data set dimensions : {}".format(dataset.shape))

dataset.groupby('diagnosis').size()

#Visualization of data
dataset.groupby('diagnosis').hist(figsize=(12, 12))

dataset.isnull().sum()
dataset.isna().sum()

dataframe = pd.DataFrame(Y)

#Encoding categorical data values 
from sklearn.preprocessing import LabelEncoder
labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_tranSet, X_testSet, Y_tranSet, Y_testSet = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
stdScl = StandardScaler()
X_tranSet = stdScl.fit_transform(X_tranSet)
X_testSet = stdScl.transform(X_testSet)
 
def perdict_Percent_Model(modName):
  Y_pred = modelLoader.predict(X_testSet)             #predicting the Test set results
  from sklearn.metrics import confusion_matrix    #Creating the confusion Matrix
  confMatx = confusion_matrix(Y_testSet, Y_pred)
  print("\n\n\n" + modName + "\n\nAccuracy : {}".format(((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0])))
  print("Sensitivity : {}".format(((confMatx [1, 1]  *100 )/ (confMatx [1, 1] + confMatx [1, 0]))))
  print("Specificity : {}".format(((confMatx [0, 0]  *100 )/ (confMatx [0, 0] + confMatx [0, 1]))))
  print("  [0, 0] :  " + str(confMatx [0, 0])+ "  [0, 1] :  " + str(confMatx [0, 1])+ "  [1, 0] :  " + str(confMatx [1, 0])+ "  [1, 1] :  " + str(confMatx [1, 1]))


#Fitting the Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
modelLoader = LogisticRegression(random_state = 0)
modelLoader.fit(X_tranSet, Y_tranSet)
perdict_Percent_Model("Logistic Regression Algorithm")


#Fitting K-NN Algorithm
from sklearn.neighbors import KNeighborsClassifier
modelLoader = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
modelLoader.fit(X_tranSet, Y_tranSet)
perdict_Percent_Model("K-NN Algorithm")


#Fitting SVM
from sklearn.svm import SVC
modelLoader = SVC(kernel = 'linear', random_state = 0)
modelLoader.fit(X_tranSet, Y_tranSet) 
perdict_Percent_Model("SVM")


#Fitting K-SVM
from sklearn.svm import SVC
modelLoader = SVC(kernel = 'rbf', random_state = 0)
modelLoader.fit(X_tranSet, Y_tranSet)
perdict_Percent_Model("K-SVM")


#Fitting Naive_Bayes
from sklearn.naive_bayes import GaussianNB
modelLoader = GaussianNB()
modelLoader.fit(X_tranSet, Y_tranSet)
perdict_Percent_Model("Naive_Bayes")


#Fitting Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
modelLoader = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
modelLoader.fit(X_tranSet, Y_tranSet)
perdict_Percent_Model("Decision Tree Algorithm")


#Fitting Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
modelLoader = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
modelLoader.fit(X_tranSet, Y_tranSet)
perdict_Percent_Model("Random Forest Classification")
