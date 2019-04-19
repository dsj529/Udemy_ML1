'''
Created on Apr 18, 2019

@author: dsj529
'''
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import numpy as np


dataset = datasets.load_iris()

features = dataset.data 
target = dataset.target 

feature_train, feature_test, target_train, target_test =\
    train_test_split(features, target, test_size=.2)

grid_params = {'criterion': ['entropy', 'gini'],
               'max_depth': np.arange(1, 10)}

tree = GridSearchCV(DecisionTreeClassifier(), grid_params, cv=5)
tree.fit(feature_train, target_train)
tree_predictions = tree.predict_proba(feature_test)[:, 1]

print("Best parameter with Grid Search: ", tree.best_params_)
