'''
Created on Apr 25, 2019

@author: dsj529

code has been rewritten to pull from quandl API.
comparison of multiple predictive algorithms for stock market price applications
'''
import datetime

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
import quandl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

import numpy as np
import pandas as pd


def makeANN():
    model = Sequential()
    # first parameter is output dimension
    model.add(Dense(50, input_dim=5, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(lr=0.005)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

def makeRNN():
    model = Sequential()
    # first parameter is output dimension
    model.add(LSTM(50, return_sequences=True, input_shape=(5, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    optimizer = Adam(lr=0.005)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

def get_data(tickers, start_date='2010-07-01', end_date='2019-03-01', lags=5):
    quandl.read_key('../data/quandl.key') # personal key to quandl API
    
    df = quandl.get_table('WIKI/PRICES', ticker='AAPL',
                          qopts={'columns': ['date', 'adj_close']},
                          date={'gte': start_date, 'lte': end_date},
                          paginate=True)
    df.date = pd.to_datetime(df.date)
    df['today'] = df['adj_close']
    df['lag_00'] = df['today'].pct_change()*100
    df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
    df.drop('date', axis=1, inplace=True)

    for i in range(lags):
        df['lag_{:02}'.format(i+1)] = df['today'].shift(i+1).pct_change()*100
    df['direction'] = np.where(df['lag_00']>=0, 1, 0) 
    
    df = df.dropna()
    return df

data = get_data('AAPL')
# print(data.head(10))
split_date = datetime.datetime(2017,6,1)

X = data[['lag_01', 'lag_02', 'lag_03', 'lag_04', 'lag_05']]
y = data[['direction']]

X_train = X[X.index < split_date]
X_lstm_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X[X.index >= split_date]
X_lstm_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
y_nn_train = y[y.index < split_date]
y_train = y_nn_train.values.ravel()
y_test = y[y.index >= split_date].values.ravel()

# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
clfs = [LogisticRegression(solver='lbfgs'),
        KNeighborsClassifier(n_neighbors=300),
        LinearSVC(),
        SVC(C=1000000.0, gamma=0.0001, kernel='rbf'),
        RandomForestClassifier(n_estimators=250,
                               criterion='entropy',
                               max_features='sqrt',
                               n_jobs=-1),
        KerasClassifier(build_fn=makeANN, epochs=1500, batch_size=20, verbose=0),
        KerasClassifier(build_fn=makeRNN, epochs=750, batch_size=20, verbose=0)]
names = ['Logistic Regression',
         'k-Neighbors',
         'SVM (Linear kernel)',
         'SVM (RBF kernel)',
         'Random Forest',
         'MLP ANN',
         'LSTM RNN']

for name, clf in zip(names, clfs):
    model_X_train = X_train
    model_X_test = X_test
    if name in ['MLP ANN', 'LSTM RNN']:
        if name == 'LSTM RNN':
            model_X_train = X_lstm_train
            model_X_test = X_lstm_test
        clf.fit(model_X_train, y_nn_train)
    else:
        clf.fit(model_X_train, y_train)
    preds = clf.predict(model_X_test)
    print('\n\t Model: {}'.format(name))
    print(confusion_matrix(preds, y_test))
    print(classification_report(preds, y_test))
    print('{:.3%}'.format(accuracy_score(preds, y_test)))