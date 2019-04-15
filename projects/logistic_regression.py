'''
Created on Apr 14, 2019

@author: dsj529
'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


data = pd.read_csv('../data/Datasets/credit_data.csv')
# print(data.head())
# print(data.describe())
# print(data.corr())

features = data[['income', 'age', 'loan']]
target = data.default

feature_train, feature_test, target_train, target_test =\
    train_test_split(features, target, test_size=0.3)

model = LogisticRegression(solver='lbfgs')
model.fit = model.fit(feature_train, target_train)
predictions = model.fit.predict(feature_test)
print('results via test-train split:')
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))

scores = cross_val_score(model, features, target, cv=10, n_jobs=-1)
preds = cross_val_predict(model, features, target, cv=10, n_jobs=-1)

print('\nresults via cross-validation:')
print('\t average score: {}'.format(np.mean(scores)))
print('\t accuracy: {}'.format(accuracy_score(target,preds)))