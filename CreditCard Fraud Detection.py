# -*- coding: utf-8 -*-
"""
Spyder Editor

Credit Card Fraud Detection Project
Author: Radhika

"""

"Importing the Data"
import pandas as pd
train_df=pd.read_csv('C:/Radhi Masters/My Preparations/Kaggle Projects/creditcard.csv/creditcard.csv')
X=train_df.drop(columns={'Class'})
y=train_df['Class']

"Splitting data into train and test datasets"
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
y_test=y_test.ravel()
y_train=y_train.ravel()

X.info()

'Finding correlation'
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,14))
corr=X.corr()
sns.heatmap(corr)


'Logisting Regression Model'
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

import seaborn as sns
sns.heatmap(cm,annot=True)
from sklearn.metrics import accuracy_score
print('Logistic Regression:',accuracy_score(y_test,y_pred))

'Printing accuracy of the model'

from sklearn.metrics import f1_score,precision_score,recall_score
print('logistic regression:',f1_score(y_test,y_pred))
print('f1_score',f1_score(y_test,y_pred))
print('precision_score',precision_score(y_test,y_pred))
print('recall score',recall_score(y_test,y_pred))

'Naive bays Model'
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred2=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2=confusion_matrix(y_test,y_pred2)
import seaborn as sns
sns.heatmap(cm2,annot=True)
'Printing model score'
from sklearn.metrics import accuracy_score
print('naive byes',accuracy_score(y_test,y_pred2))
from sklearn.metrics import f1_score,precision_score,recall_score
print('f1_score',f1_score(y_test,y_pred2))
print('precision_score',precision_score(y_test,y_pred2))
print('recall_score',  recall_score(y_test,y_pred2))

'Decission Tree Model'
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred3=classifier.predict(X_test)
print(y_pred3)
from sklearn.metrics import confusion_matrix
cm3=confusion_matrix(y_test,y_pred3)
import seaborn as sns
sns.heatmap(cm3,annot=True)
from sklearn.metrics import accuracy_score
print('decision tree',accuracy_score(y_test,y_pred3))
from sklearn.metrics import f1_score,precision_score,recall_score
print('f1_score',f1_score(y_test,y_pred3))
print('precisiion_score',precision_score(y_test,y_pred3))
print('recall_score',recall_score(y_test,y_pred3))

'Radom Forest Classifier'
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred4=classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print('accuracy_score',accuracy_score(y_test,y_pred4))
print(y_pred4)
from sklearn.metrics import f1_score,precision_score,recall_score
print('f1_score',f1_score(y_test,y_pred4))
print('precision_score',precision_score(y_test,y_pred4))
print('recall_score',recall_score(y_test,y_pred4))

