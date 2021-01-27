import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import *

## fonction du fichier LinearStudy

def eval_metric(title, yTestTrue, yTestPred, yTrainTrue, yTrainPred, numpy=0, neg=0, scientific=0):
    listmetric = []
    listmetric.append(title)
    
    if scientific == 0:
        listmetric.append(round(mean_squared_error(yTestTrue, yTestPred),3))
    else:
        listmetric.append('{:0.3e}'.format(mean_squared_error(yTestTrue, yTestPred)))
    
    listmetric.append(round(r2_score(yTestTrue, yTestPred),3))
    if neg ==0:
        listmetric.append(round(mean_squared_log_error(yTestTrue, yTestPred),5))
    else:
        listmetric.append('NaN')
    
    listmetric.append(round(mean_squared_error(yTestTrue, yTestPred) / mean_squared_error(yTrainTrue, yTrainPred),3))
    if numpy == 0:
        return listmetric
    else:
        return np.array(listmetric)


def outliersTrainTest(y_train, y_test):
    outliers_train = y_train[y_train == 0].index
    outliers_test  =   y_test[y_test == 0].index
    return outliers_train, outliers_test

def removeOutliersXy(X_train, X_test, y_train, y_test, outliers_train, outliers_test):
    X_train_woOut = X_train.drop(outliers_train)
    X_test_woOut  =  X_test.drop(outliers_test)
    y_train_woOut = y_train.drop(outliers_train)
    y_test_woOut  =  y_test.drop(outliers_test)
    return X_train_woOut, X_test_woOut, y_train_woOut, y_test_woOut 

def applyScalerTrainTest(scaler, X_train, X_test):
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def applyLogYTrainTest(y_train, y_test):
    y_train_log = np.log(1 + y_train)
    y_test_log  = np.log(1 + y_test)
    return y_train_log, y_test_log 

def linearRegTrainTestPred(X_train, X_test, y_train):
    ## modèle de régression linéaire
    lr = linear_model.LinearRegression()
    ## entraînement sur les données
    lr.fit(X_train, y_train)              
    
    ## prédictions 
    y_train_pred = lr.predict(X_train)  ## sur jeu d'entraînement
    y_test_pred  = lr.predict(X_test)   ## sur jeu de test
    
    return y_train_pred, y_test_pred

def yTruePred(y_true, y_pred, nameTrue = 'true', namePred = 'pred',
             error = True, relativeError = True):
    df = pd.DataFrame({nameTrue : y_true,
                       namePred : y_pred})
    
    if error: 
        df['error'] = df[namePred] - df[nameTrue]
    if relativeError:
        df['relative_error'] = df[namePred] / df[nameTrue]
    
    return df
