# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import sys

    
data = pd.read_csv('../data/processed/AAPL_historical_stock_prices_preprocessed.csv')

dataset_train=data.iloc[0:930,1:2]
dataset_test=data.iloc[930:,1:2]
training_set = data.iloc[0:930, 3:4].values
testing_set=data.iloc[930:,3:4].values

data.head()

print(data.head())
data = data.iloc[::-1]

plt.figure(figsize = (18,9))
plt.plot(range(data.shape[0]),(data['Close']))
plt.xticks(range(0,data.shape[0],500),data['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price',fontsize=18)
plt.show()

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
len(training_set_scaled)
X_train = []
y_train = []
for i in range(10,930):
    X_train.append(training_set_scaled[i-10:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from tensorflow.keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 75, return_sequences = True, input_shape = (X_train.shape[1], 1), activation='relu'))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(GRU(units = 50, return_sequences = True))
regressor.add(Dropout(0.1))

regressor.add(SimpleRNN(units = 75))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.summary()

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 200, batch_size = 64)

real_stock_price = testing_set
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 10:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# Prepare input data for the next week's prediction
next_week_inputs = inputs[-10:]  # Select the last 10 data points
next_week_predictions = []  # Store predictions for the next week

# Number of days to predict (assuming a 5-day trading week)
days_to_predict = 5

for _ in range(days_to_predict):
    # Reshape input data
    next_week_inputs = np.reshape(next_week_inputs, (1, next_week_inputs.shape[0], 1))

    # Predict the next day's stock price
    next_day_prediction = regressor.predict(next_week_inputs)

    # Store the prediction
    next_week_predictions.append(next_day_prediction[0, 0])

    # Update input data for the next prediction
    next_week_inputs = np.append(next_week_inputs[0][1:], next_day_prediction)

# Inverse transform the predictions
next_week_predictions = sc.inverse_transform(np.array(next_week_predictions).reshape(-1, 1))
print(f"Next week's stock price predictions: {next_week_predictions.flatten()}")

plt.plot(next_week_predictions, color='blue', label='Next Week Predictions')
plt.title('Next Week Stock Price Prediction')
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


def AAPL_predictions():
    return next_week_predictions.flatten()

