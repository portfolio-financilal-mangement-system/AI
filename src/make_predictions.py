import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
with open('AAPL_model.pkl', 'rb') as f:
    regressor = pickle.load(f)

# Load the preprocessed data for prediction
data = pd.read_csv('../data/processed/AAPL_historical_stock_prices_preprocessed.csv')

# Scale the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set = sc.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare input data for the next week's prediction
inputs = training_set[-10:].reshape((1, 10, 1))

# Predict the next 5 days' stock prices
next_week_predictions = []
for _ in range(5):
    # Predict the next day's stock price
    next_day_prediction = regressor.predict(inputs)
    
    # Store the prediction
    next_week_predictions.append(next_day_prediction[0, 0])
    
    # Update input data for the next prediction
    inputs = np.append(inputs[0][1:], next_day_prediction).reshape((1, 10, 1))

# Inverse transform the predictions
next_week_predictions = sc.inverse_transform(np.array(next_week_predictions).reshape(-1, 1))

# Print the predictions
print("Next 5 Days' Predictions:")
for i, prediction in enumerate(next_week_predictions):
    print(f"Day {i+1}: {prediction[0]}")
