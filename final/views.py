from django.shortcuts import render
import numpy as np
import pandas as pd
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

def stats(request):
    print("stats")
    data = pd.read_csv('static/data.csv')
    X = data.iloc[:, 2:14].values
    Y = data.iloc[:, 1].values

    labEnc_Y = LabelEncoder()
    Y = labEnc_Y.fit_transform(Y)
    
    X_tranSet, X_testSet, Y_tranSet, Y_testSet = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    stdScl = StandardScaler()
    X_tranSet = stdScl.fit_transform(X_tranSet)
    X_testSet = stdScl.transform(X_testSet)


    #Fitting the Logistic Regression Algorithm to the Training Set
    modelLoader = LogisticRegression(random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a1= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    s1= (((confMatx [1, 1]  *100 )/ (confMatx [1, 1] + confMatx [1, 0])))
    p1=(((confMatx [0, 0]  *100 )/ (confMatx [0, 0] + confMatx [0, 1])))
    g1= ((a1+s1+p1)/3)

    #Fitting K-NN Algorithm
    modelLoader = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a2= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    s2= (((confMatx [1, 1]  *100 )/ (confMatx [1, 1] + confMatx [1, 0])))
    p2=(((confMatx [0, 0]  *100 )/ (confMatx [0, 0] + confMatx [0, 1])))
    g2= ((a2+s2+p2)/3)
    
    #Fitting SVM
    modelLoader = SVC(kernel = 'linear', random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a3= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    s3= (((confMatx [1, 1]  *100 )/ (confMatx [1, 1] + confMatx [1, 0])))
    p3=(((confMatx [0, 0]  *100 )/ (confMatx [0, 0] + confMatx [0, 1])))
    g3= ((a3+s3+p3)/3)

    #Fitting K-SVM
    modelLoader = SVC(kernel = 'rbf', random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a4= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    s4= (((confMatx [1, 1]  *100 )/ (confMatx [1, 1] + confMatx [1, 0])))
    p4=(((confMatx [0, 0]  *100 )/ (confMatx [0, 0] + confMatx [0, 1])))
    g4= ((a4+s4+p4)/3)

    #Fitting Naive_Bayes
    modelLoader = GaussianNB()
    modelLoader.fit(X_tranSet, Y_tranSet)   
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a5= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    s5= (((confMatx [1, 1]  *100 )/ (confMatx [1, 1] + confMatx [1, 0])))
    p5=(((confMatx [0, 0]  *100 )/ (confMatx [0, 0] + confMatx [0, 1])))
    g5= ((a5+s5+p5)/3)

    #Fitting Decision Tree Algorithm
    modelLoader = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a6= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    s6= (((confMatx [1, 1]  *100 )/ (confMatx [1, 1] + confMatx [1, 0])))
    p6=(((confMatx [0, 0]  *100 )/ (confMatx [0, 0] + confMatx [0, 1])))
    g6= ((a6+s6+p6)/3)

    #Fitting Random Forest Classification Algorithm
    modelLoader = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a7= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    s7= (((confMatx [1, 1]  *100 )/ (confMatx [1, 1] + confMatx [1, 0])))
    p7=(((confMatx [0, 0]  *100 )/ (confMatx [0, 0] + confMatx [0, 1])))
    g7= ((a7+s7+p7)/3)
    

    return render(request, 'stats.html',{'title':'Model Evalution','acti':'nav-acti','a1':a1,'s1':s1,'p1':p1,'g1':g1,
'a2':a2,'s2':s2,'p2':p2,'g2':g2,'a3':a3,'s3':s3,'p3':p3,'g3':g3,
'a4':a4,'s4':s4,'p4':p4,'g4':g4,'a5':a5,'s5':s5,'p5':p5,'g5':g5,
'a6':a6,'s6':s6,'p6':p6,'g6':g6,'a7':a7,'s7':s7,'p7':p7,'g7':g7})
   
def predict(request):
    print("predict")
    data = pd.read_csv('static/data.csv')
    X = data.iloc[:, 2:14].values
    Y = data.iloc[:, 1].values

    labEnc_Y = LabelEncoder()
    Y = labEnc_Y.fit_transform(Y)
    
    X_tranSet, X_testSet, Y_tranSet, Y_testSet = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    stdScl = StandardScaler()
    X = stdScl.fit_transform(X)
    X_tranSet = stdScl.transform(X_tranSet)
    X_testSet = stdScl.transform(X_testSet)

    modelLoader = LogisticRegression(random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a1= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    
    #Fitting K-NN Algorithm
    modelLoader = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a2= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    
    #Fitting SVM
    modelLoader = SVC(kernel = 'linear', random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a3= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    
    #Fitting K-SVM
    modelLoader = SVC(kernel = 'rbf', random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a4= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    
    #Fitting Naive_Bayes
    modelLoader = GaussianNB()
    modelLoader.fit(X_tranSet, Y_tranSet)   
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a5= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    
    #Fitting Decision Tree Algorithm
    modelLoader = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a6= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    
    #Fitting Random Forest Classification Algorithm
    modelLoader = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    modelLoader.fit(X_tranSet, Y_tranSet)
    Y_pred = modelLoader.predict(X_testSet)
    confMatx = confusion_matrix(Y_testSet, Y_pred)
    a7= (((confMatx [0, 0] + confMatx [1, 1]) *100 )/ (confMatx [0, 0] + confMatx [1, 1] + confMatx [0, 1] + confMatx [1, 0]))
    
    max_ac = max(a1,a2,a3,a4,a5,a6,a7)
    print (max_ac)
    value = ''

    if request.method == 'POST':
        mrad = float(request.POST['mradius'])
        mtex = float(request.POST['mtexture'])
        msmt = float(request.POST['msmoothness'])
        mcom = float(request.POST['mcompactness'])

        serad = float(request.POST['seradius'])
        setex = float(request.POST['setexture'])
        sesmt = float(request.POST['sesmoothness'])
        secom = float(request.POST['secompactness'])

        wrad = float(request.POST['wradius'])
        wtex = float(request.POST['wtexture'])
        wsmt = float(request.POST['wsmoothness'])
        wcom = float(request.POST['wcompactness'])
        inp = np.array(
            (mrad, mtex, msmt, mcom, serad, setex, sesmt, secom, wrad, wtex, wsmt, wcom,)
        ).reshape(1, 12)
        inp = stdScl.transform(inp)
        pred = 5
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
    
    return render(request,'predict.html', {'value':value, 'title':'Try to Predict !','acti':'nav-acti'})
   
def home(request):
    print("Home")
    return render(request, 'home.html', {'title':'Home','acti':'nav-acti'})