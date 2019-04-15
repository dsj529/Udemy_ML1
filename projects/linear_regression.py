'''
Created on Apr 14, 2019

@author: dsj529
'''
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def is_outlier(series, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(series.shape) == 1:
        points = series[:,None]
    median = np.median(series, axis=0)
    diff = np.sum((series - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def eval_model(model, X, y, preds):
    y_med = (y.max() - y.min())/2
    model_rmse = math.sqrt(mean_squared_error(y, preds))
    print('Median Y: {:,}'.format(y_med))
    print('RMSE: {:,.3f}, ({:.2%})'.format(model_rmse, model_rmse/y_med))
    print('R2: {:.6}'.format(model.score(X,y)))
    
    print('b1: {}, b0: {}'.format(model.coef_[0], model.intercept_[0]))
    
data = pd.read_csv('../data/Datasets/house_prices.csv')
size = data['sqft_living']
price = data['price']

X1 = np.array(size).reshape(-1,1)
y1 = np.array(price).reshape(-1,1)

mask = ~is_outlier(X1)
X2 = X1[mask]
y2 = y1[mask]

model1 = LinearRegression()
model1.fit(X1,y1)
preds1 = model1.predict(X1)

model2 = LinearRegression()
model2.fit(X2,y2)
preds2 = model2.predict(X2)
## evaluate the models
eval_model(model1, X1, y1, preds1)
eval_model(model2, X2, y2, preds2)

## visualize the data and model
fig, axs = plt.subplots(1, 2, figsize=(12,9))
axs[0].scatter(X1, y1, color='green')
axs[0].plot(X1, preds1, color='black')
axs[0].set_title('Linear Regression')
axs[0].set_xlabel('Size')
axs[0].set_ylabel('Price')

axs[1].scatter(X2, y2, color='green')
axs[1].plot(X2, preds2, color='black')
axs[1].set_title('Linear Regression with outlier removal')
axs[1].set_xlabel('Size')
axs[1].set_ylabel('Price')

plt.show()

## I was definitely not expecting the outlier removal to degrade the predictive power of the model.
## This was definitely an interesting lesson in how data behaves.
