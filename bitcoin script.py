# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:36:55 2022

@author: jacob
"""

import pandas as pd
import numpy as np
#data is from https://www.cryptodatadownload.com/
BTC_price = pd.read_csv('Gemini_BTCUSD_day.csv',header=1)
BTC_price.head()

#drop the first column and reverse order
BTC_price = BTC_price.iloc[1:,:]
BTC_price = BTC_price.iloc[::-1]
BTC_price['Date'] = pd.to_datetime(BTC_price['Date'])
BTC_price.set_index('Date', inplace=True)
BTC_price.tail()

import matplotlib.pyplot as plt
ax = BTC_price.plot(y= 'Close', figsize=(12,6), legend=True, grid=True, use_index=True)
plt.show()

#select duration
initial_date = '2017-01-01'
finish_date = '2017-12-01'
BTC_price_time = BTC_price[initial_date:finish_date]

#Helper function to extract a rolling window
#We need to specify both historical window and future window
#to make sure that our matrix is consistent
def GetWindow(x,h_window =30,f_window=10):

    # First window
    X = np.array(x.iloc[:h_window,]).reshape(1,-1)
   
    # Append next window
    for i in range(1,len(x)-h_window+1):
        x_i = np.array(x.iloc[i:i+h_window,]).reshape(1,-1)
        X = np.append(X,x_i, axis=0)
        
    # Cut the end that we can't use to predict future price
    rolling_window = (pd.DataFrame(X)).iloc[:-f_window,]
    return rolling_window

#input = panda, historical window, future window
def GetNextMean(x,h_window=30,f_window=10):
    return pd.DataFrame((x.rolling(f_window).mean().iloc[h_window+f_window-1:,]))

BTC_price_time['Close'].head(10)

GetWindow(BTC_price_time.loc[:,'Close'].head(10), h_window = 5, f_window =2)

GetNextMean(BTC_price_time.loc[:,'Close'].head(10), h_window = 5, f_window =2)

#Function add time to the data set
def AddTime(X):
    t = np.linspace(0,1,len(X))
    return np.c_[t, X]

#Function for Lead lag transform
def Lead(X):
    
    s = X.shape
    x_0 = X[:,0]
    Lead = np.delete(np.repeat(x_0,2),0).reshape(-1,1)
     
    for j in range(1,s[1]):
        x_j = X[:,j]
        x_j_lead = np.delete(np.repeat(x_j,2),0).reshape(-1,1)
        Lead = np.concatenate((Lead,x_j_lead), axis =1)
     
    return Lead

#Function for Lead lag transform
def Lag(X):
    
    s = X.shape
    x_0 = X[:,0]
    Lag = np.delete(np.repeat(x_0,2),-1).reshape(-1,1)
  
    for j in range(1,s[1]):
        x_j = X[:,j]
        x_j_lag  = np.delete(np.repeat(x_j,2),-1).reshape(-1,1)
        Lag = np.concatenate((Lag,x_j_lag), axis = 1)
        
    return Lag

import esig.tosig as ts

# We use only close price
close_price = BTC_price_time.loc[:,'Close']
h_window = 30
f_window = 10
sig_level = 2

# mean next price
y = GetNextMean(close_price, h_window = h_window , f_window = f_window)

# normal window features
X_window = AddTime(GetWindow(close_price, h_window = h_window, f_window = f_window))
X_window = pd.DataFrame(X_window)


# signature features
#Consider only area that has at least f_window future prices left
close_price_slice = close_price.iloc[0:(len(close_price)-(f_window))]
close_price_array = np.array(close_price_slice).reshape(-1,1)
lag = Lag(close_price_array)
lead = Lead(AddTime(close_price_array))
#concatenate everything to get a datastream
stream = np.concatenate((lead,lag), axis = 1)
X_sig = [ts.stream2sig(stream[0:2*h_window-1], sig_level)]

for i in range(1,(len(close_price)-(f_window)-(h_window)+1)):
    stream_i = stream[2*i: 2*(i+h_window)-1]
    signature_i = [ts.stream2sig(stream_i, sig_level)]
    X_sig = np.append(X_sig, signature_i, axis=0)

X_sig = pd.DataFrame(X_sig)

y.head()
pd.DataFrame(X_window).head()
pd.DataFrame(X_sig).head()

# Split X,y into train and test region
test_len = 10
train_len = len(y) - test_len

X_train = X_sig.iloc[:train_len,]
y_train = y.iloc[:train_len,]
X_test = X_sig.iloc[train_len:,]
y_test = y.iloc[train_len:,]

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

#mean absolute percentage error
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Perform a GridsearchCV
param_search = {'alpha': [1e-5,1e-2,1e-1,2e-1, 5e-1, 1]}
myModel = GridSearchCV(estimator=Lasso(alpha = 0.1,
                                       normalize= True,
                                       max_iter = 10e5),
                        param_grid = param_search,
                        cv = TimeSeriesSplit(n_splits=5),
                        n_jobs=-1)

myModel.fit(X_train, y_train)

#Make predictions
y_train_predict = myModel.predict(X_train)
y_test_predict = myModel.predict(X_test)

#Calculate error
error_train = mean_absolute_error(y_train, y_train_predict)        
error_test = mean_absolute_error(y_test, y_test_predict)
p_error_train = mean_absolute_percentage_error(np.array(y_train).reshape(-1,1), np.array(y_train_predict).reshape(-1,1))
p_error_test = mean_absolute_percentage_error(np.array(y_test).reshape(-1,1), np.array(y_test_predict).reshape(-1,1))

print('mae_train:{0:.3f} = {1:.3f}%\n'.format(error_train, p_error_train))
print('mae_test:{0:.3f} = {1:.3f}%'.format(error_test, p_error_test))

# Plot to visualise
import matplotlib.pyplot as plt

def PlotResult(y_train, y_test, y_train_predict, y_test_predict, test_len, name):
        
        #Visualise
        plt.figure(figsize=(12, 5))
        plt.plot(y_train_predict,color='red')
        
        plt.plot(range(train_len, train_len+len(y_test)),
                 y_test_predict,
                 label='Predicted average price',
                 color='red',linestyle = '--')
        
        plt.plot(np.array((y_train).append(y_test)),
                 label='Actual average price',
                 color='green')
        
        plt.axvspan(len(y_train), len(y_train)+len(y_test),
                    alpha=0.3, color='lightgrey')
        
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc="best")
        plt.title('Predicting the mean BTC price with {}'.format(name))
        
PlotResult(y_train, y_test, y_train_predict, y_test_predict, test_len, 'Lasso + Signature features')

##XGBoost

# Split X,y into train and test region
test_len = 10
train_len = len(y) - test_len

X_train = X_window.iloc[:train_len,]
y_train = y.iloc[:train_len,]
X_test = X_window.iloc[train_len:,]
y_test = y.iloc[train_len:,]

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

#GridsearchCV    
param_search = {
    'max_depth':[4,5,6]
    ,'min_child_weight':[4,5,6]
    ,'gamma':[i/10.0 for i in range(0,5)]
}

myModel = GridSearchCV(estimator=XGBRegressor(
                        learning_rate=0.01,
                        n_estimators=500,
                        max_depth=5,
                        min_child_weight=5,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8, 
                        eval_metric ='mae',
                        reg_alpha=0.05
                        ),
                       param_grid = param_search,
                       cv = TimeSeriesSplit(n_splits=5),n_jobs=-1
                      )

#Fit model
myModel.fit(X_train, y_train)

#Make predictions
y_train_predict = myModel.predict(X_train)
y_test_predict = myModel.predict(X_test)

#Calculate error
error_train = mean_absolute_error(y_train, y_train_predict)        
error_test = mean_absolute_error(y_test, y_test_predict)
p_error_train = mean_absolute_percentage_error(np.array(y_train).reshape(-1,1), np.array(y_train_predict).reshape(-1,1))
p_error_test = mean_absolute_percentage_error(np.array(y_test).reshape(-1,1), np.array(y_test_predict).reshape(-1,1))

print('mae_train:{0:.3f} = {1:.3f}%\n'.format(error_train, p_error_train))
print('mae_test:{0:.3f} = {1:.3f}%'.format(error_test, p_error_test))

PlotResult(y_train, y_test, y_train_predict, y_test_predict, test_len, 'XGBoost')

##Experiment with other period of time

#select duration
initial_date = '2018-01-01'
finish_date = '2018-11-01'

