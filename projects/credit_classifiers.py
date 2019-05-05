'''
Created on Apr 19, 2019

@author: dsj529
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd


data = pd.read_csv('../data/Datasets/credit_data.csv')
features = data[['income', 'age', 'loan']]
target = data.default

feature_train, feature_test, target_train, target_test =\
    train_test_split(features, target, test_size=0.3)
    
estimators = [GaussianNB(),
              LogisticRegression(solver='lbfgs'),
              KNeighborsClassifier(n_neighbors=20),
              RandomForestClassifier(criterion='entropy', n_estimators=2500, max_features='sqrt')]
names = ['Naive_Bayes', 'Logistic Regression', 'k-Neighbors Classifier', 'Random Forest']

for clf, name in zip(estimators, names):
    model = clf.fit(feature_train, target_train)
    preds = clf.predict(feature_test)
    print('\nResults for {} model:'.format(name))
    print('{}'.format(confusion_matrix(target_test, preds)))
    print('{}'.format(classification_report(target_test, preds)))
    print('Overall accuracy: {:.4%}'.format(accuracy_score(target_test, preds)))