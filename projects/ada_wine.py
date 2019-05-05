'''
Created on Apr 22, 2019

@author: dsj529
'''
from nltk.classify.decisiontree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def make_df():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-{}.csv"
    names = ['fixed_acid', 'volatile_acid',
             'citric_acid', 'resid_sugar',
             'chlorides', 'free_SO2',
             'total_SO2', 'density',
             'pH', 'sulphates', 'quality']
    
    df = pd.read_csv(url.format('red'), sep=';')
    df = df.append(pd.read_csv(url.format('white'), sep=';'))
    return df

dataset = make_df()#.values
features = dataset.iloc[:, :-1].values
target1 = dataset.iloc[:, -1].values
target2 = np.where(target1 > 7, 1, 0)
split1 = train_test_split(features, target1, test_size=0.3)
split2 = train_test_split(features, target2, test_size=0.3)

for X_train, X_test, y_train, y_test in [split1, split2]:
    param_grid = {'n_estimators': [50, 100, 250, 500, 1000],
                  'learning_rate': [0.01, 0.03, 0.1, 0.3, 1, 3]}
    grid_model = GridSearchCV(
        param_grid=param_grid,
        cv=3,
        estimator=AdaBoostClassifier())
    fitted = grid_model.fit(X_train, y_train)
    model_preds = fitted.predict(X_test)

    print(classification_report(y_test, model_preds))
    print(accuracy_score(y_test, model_preds))
    print('\n')