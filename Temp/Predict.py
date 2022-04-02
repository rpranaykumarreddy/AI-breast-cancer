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
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 2:14].values
Y = dataset.iloc[:, 1].values

#Encoding categorical data values 
labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

X_tranSet, X_testSet, Y_tranSet, Y_testSet = train_test_split(X, Y, test_size = 0.25, random_state = 0)

stdScl = StandardScaler()
X = stdScl.fit_transform(X)
X_tranSet = stdScl.transform(X_tranSet)
X_testSet = stdScl.transform(X_testSet)

modelLoader = LogisticRegression(random_state = 0)
modelLoader.fit(np.nan_to_num(X_tranSet), Y_tranSet)
Y_pred = modelLoader.predict(X_testSet)
confMatx = confusion_matrix(Y_testSet, Y_pred)
a1= accuracy_score(Y_testSet, Y_pred)*100

#Fitting K-NN Algorithm
modelLoader = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
modelLoader.fit(np.nan_to_num(X_tranSet), Y_tranSet)
Y_pred = modelLoader.predict(X_testSet)
confMatx = confusion_matrix(Y_testSet, Y_pred)
a2= accuracy_score(Y_testSet, Y_pred)*100

#Fitting SVM
modelLoader = SVC(kernel = 'linear', random_state = 0)
modelLoader.fit(np.nan_to_num(X_tranSet), Y_tranSet)
Y_pred = modelLoader.predict(X_testSet)
confMatx = confusion_matrix(Y_testSet, Y_pred)
a3= accuracy_score(Y_testSet, Y_pred)*100

#Fitting K-SVM
modelLoader = SVC(kernel = 'rbf', random_state = 0)
modelLoader.fit(np.nan_to_num(X_tranSet), Y_tranSet)
Y_pred = modelLoader.predict(X_testSet)
confMatx = confusion_matrix(Y_testSet, Y_pred)
a4= accuracy_score(Y_testSet, Y_pred)*100

#Fitting Naive_Bayes
modelLoader = GaussianNB()
modelLoader.fit(np.nan_to_num(X_tranSet), Y_tranSet)   
Y_pred = modelLoader.predict(X_testSet)
confMatx = confusion_matrix(Y_testSet, Y_pred)
a5= accuracy_score(Y_testSet, Y_pred)*100

#Fitting Decision Tree Algorithm
modelLoader = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
modelLoader.fit(np.nan_to_num(X_tranSet), Y_tranSet)
Y_pred = modelLoader.predict(X_testSet)
confMatx = confusion_matrix(Y_testSet, Y_pred)
a6= accuracy_score(Y_testSet, Y_pred)*100

#Fitting Random Forest Classification Algorithm
modelLoader = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
modelLoader.fit(np.nan_to_num(X_tranSet), Y_tranSet)
Y_pred = modelLoader.predict(X_testSet)
confMatx = confusion_matrix(Y_testSet, Y_pred)
a7= accuracy_score(Y_testSet, Y_pred)*100


max_ac = max(a1,a2,a3,a4,a5,a6,a7)
print("\n\nmaximum Accuracy   :   " + str(max_ac))
value = ''

inp = [[7.76, 24.54, 0.05263, 0.04362, 0.3857, 1.428, 0.007189, 0.00466, 9.456, 30.37, 0.08996, 0.06444]]


#Feature Scaling
stdScl = StandardScaler()
X = stdScl.fit_transform(X)
inp = stdScl.transform(inp)

if (max_ac == a1):
    print("a1")
    model = LogisticRegression(random_state = 0)
    model.fit(np.nan_to_num(X), Y)
    pred = model.predict(inp)
elif (max_ac == a2):
    print("a2")
    model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    model.fit(np.nan_to_num(X), Y)
    pred = model.predict(inp)
elif (max_ac == a3):
    print("a3")
    model = SVC(kernel = 'linear', random_state = 0)
    model.fit(np.nan_to_num(X), Y)
    pred = model.predict(inp)
elif (max_ac == a4):
    print("a4")
    model = SVC(kernel = 'rbf', random_state = 0)
    model.fit(np.nan_to_num(X), Y)
    pred = model.predict(inp)
elif (max_ac == a5):
    print("a5")
    model = GaussianNB()
    model.fit(np.nan_to_num(X), Y)
    pred = model.predict(inp)
elif (max_ac == a6):
    print("a6")
    model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    model.fit(np.nan_to_num(X), Y)
    pred = model.predict(inp)
elif (max_ac == a7):
    print("a7")
    model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    model.fit(np.nan_to_num(X), Y)
    pred = model.predict(inp)

print(pred[0])

if int(pred[0]) == 1:
    value = 'Maligant'
elif int(pred[0]) == 0:
    value = "Benign"    

print(value)