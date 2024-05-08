import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
import pickle

def train_model(company):
    # Get the path to the data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed'))
    data_file = os.path.join(data_dir, f'{company}_historical_stock_prices_preprocessed.csv')


    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return

    # Load the data for the specified company
    data = pd.read_csv(data_file)

    # Scale the data
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set = sc.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare the training data
    X_train = []
    y_train = []
    for i in range(10, len(training_set)):
        X_train.append(training_set[i-10:i, 0])
        y_train.append(training_set[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the model
    regressor = Sequential()
    regressor.add(LSTM(units=75, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='relu'))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(GRU(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(SimpleRNN(units=75))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))

    # Compile the model
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    regressor.fit(X_train, y_train, epochs=200, batch_size=64)

    # Save the model with a different file name for each company
    model_file = os.path.join(os.path.dirname(__file__), f'{company}_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(regressor, f)


train_model('AAPL')
train_model('MSFT')
train_model('GOOGL')