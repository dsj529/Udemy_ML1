'''
Created on Apr 14, 2019

@author: dsj529
'''
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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
    
model = GaussianNB().fit(feature_train, target_train)
preds = model.predict(feature_test)

print(confusion_matrix(target_test, preds))
print('Accuracy score: {:.3%}'.format(accuracy_score(target_test, preds)))