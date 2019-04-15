'''
Created on Apr 14, 2019

@author: dsj529
'''
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre


data = pd.read_csv('../data/Datasets/credit_data.csv')
# print(data.head())
# print(data.describe())
# print(data.corr())

features = data[['income', 'age', 'loan']]
target = data.default

features = skpre.MinMaxScaler().fit_transform(features)

feature_train, feature_test, target_train, target_test =\
    train_test_split(features, target, test_size=0.3)
    
model = KNeighborsClassifier(n_neighbors=20)
fit_model = model.fit(feature_train, target_train)
preds = fit_model.predict(feature_test)

cv_scores = []

for k in range(1,101):
    print('now evaluating k={}'.format(k))
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, features, target, cv=10, scoring='accuracy')
    cv_scores.append(score.mean())
    
print('optimal neighborhood: {}'.format(np.argmax(cv_scores)))


print(confusion_matrix(target_test, preds))
print('Accuracy score: {:.3%}'.format(accuracy_score(target_test, preds)))