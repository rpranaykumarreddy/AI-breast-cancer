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
from sklearn.metrics import confusion_matrix
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



#Encoding categorical data values 
labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
X_tranSet, X_testSet, Y_tranSet, Y_testSet = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
stdScl = StandardScaler()
X_tranSet = stdScl.fit_transform(X_tranSet)
X_testSet = stdScl.transform(X_testSet)
 
def perdict_Percent_Model(modName):
  modelLoader.fit(np.nan_to_num(X_tranSet), Y_tranSet)
  Y_pred = modelLoader.predict(X_testSet)         #Creating the confusion Matrix
  confMatx = confusion_matrix(Y_testSet, Y_pred)
  print("\n\n\n" + modName + "\n\nAccuracy : {}".format(accuracy_score(Y_testSet, Y_pred)*100))
  print("Sensitivity : {}".format(((confMatx [1, 1]  *100 )/ (confMatx [1, 1] + confMatx [1, 0]))))
  print("Specificity : {}".format(((confMatx [0, 0]  *100 )/ (confMatx [0, 0] + confMatx [0, 1]))))
  print("  [0, 0] :  " + str(confMatx [0, 0])+ "  [0, 1] :  " + str(confMatx [0, 1])+ "  [1, 0] :  " + str(confMatx [1, 0])+ "  [1, 1] :  " + str(confMatx [1, 1]))


#Fitting the Logistic Regression Algorithm to the Training Set
modelLoader = LogisticRegression(random_state = 0)
perdict_Percent_Model("Logistic Regression Algorithm")

#Fitting K-NN Algorithm
modelLoader = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
perdict_Percent_Model("K-NN Algorithm")

#Fitting SVM
modelLoader = SVC(kernel = 'linear', random_state = 0)
perdict_Percent_Model("SVM")

#Fitting K-SVM
modelLoader = SVC(kernel = 'rbf', random_state = 0)
perdict_Percent_Model("K-SVM")

#Fitting Naive_Bayes
modelLoader = GaussianNB()
perdict_Percent_Model("Naive_Bayes")

#Fitting Decision Tree Algorithm
modelLoader = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
perdict_Percent_Model("Decision Tree Algorithm")

#Fitting Random Forest Classification Algorithm
modelLoader = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
perdict_Percent_Model("Random Forest Classification")

print("\n\nAccuracy of total dataset i.e ")
print("Cancer data set dimensions : {}".format(dataset.shape))
