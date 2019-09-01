# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:11:09 2019

@author: Josue Prieto
"""

import pandas as pd 
import matplotlib.pyplot as plt
import pc
df = pd.read_csv('data.csv')

# The exploration begins 


print(df.info())

print(df.isnull().sum())



from sklearn import preprocessing

X = df.drop(columns=['Personal Loan','ID','ZIP Code'])
y = df['Personal Loan']


"""
scaler = preprocessing.StandardScaler()


columns =X.columns.tolist()

X = scaler.fit_transform(X)

X = pd.DataFrame(X)
X.set_axis(columns,axis='columns')
"""
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, recall_score
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)


"""
Random Forest

"""

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=20).fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, rfc_pred))
print('Recall', recall_score(y_test, rfc_pred, average='macro'))

cf_rf = confusion_matrix(y_test,rfc_pred)
print(cf_rf)

clases = ['No incendio','Incendio']
pc.print_confusion_matrix(cf_rf,clases)