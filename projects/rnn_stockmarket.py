import os

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def init_data(window=40):
    prices_dataset_train = pd.read_csv('../data/Datasets/SP500_train.csv')
    prices_dataset_test = pd.read_csv('../data/Datasets/SP500_test.csv')

    #we are after a given column in the dataset
    training_set = prices_dataset_train.iloc[:,5:6].values
    test_set = prices_dataset_test.iloc[:,5:6].values

    test_data = np.vstack((training_set[-window:,:],test_set)).reshape(-1,1)
    # tack last 40 days' data from train onto the beginning of test data, 
    # reshape to make sure it's in column-vector form

    return training_set, test_set, test_data

def prep_data(data, window=40):
    X_train = []
    y_train = []
    out_width = len(data) - window

    for i in range(out_width):
        X_train.append(data[i:(i - out_width), 0])
        y_train.append(data[(i - out_width + 1), 0])

    return np.array(X_train), np.array(y_train)

def make_model(shape):
    #return sequence true because we have another LSTM after this one
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(shape[1],1)))
    model.add(Dropout(0.5))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    #RMSProp is working fine with LSTM but so do ADAM optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_results(y, y_hat):
    plt.plot(y, color='blue', label='Actual S&P500 Prices')
    plt.plot(y_hat, color='green', label='LSTM Predictions')
    plt.title('S&P500 Predictions with Recurrent Neural Network')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    (train_set, test_set, test_data) = init_data()

    #we use min-max normalization to normalize the dataset
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_set = min_max_scaler.fit_transform(train_set)

    X_train, y_train = prep_data(scaled_train_set)

    # input shape for LSTM architecture
    # we have to reshape the dataset (numOfSamples,numOfFeatures,1)
    # we have 1 because we want to predict the price tomorrow (so 1 value)
    # numOfFeatures: the past prices we use as features
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    model = make_model(X_train.shape)
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    ## test the model
    test_input = min_max_scaler.transform(test_data)
    X_test, _ = prep_data(test_input)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    preds = model.predict(X_test)
    preds = min_max_scaler.inverse_transform(preds)

    plot_results(test_set, preds)

if __name__ == "__main__":
    main()
