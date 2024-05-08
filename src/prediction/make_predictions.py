
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import os

def make_predictions(company):
    # Get the absolute path to the model file for the specified company
    model_path = os.path.join(os.path.dirname(__file__), f'{company}_model.pkl')
    
    try:
        # Load the saved model
        with open(model_path, 'rb') as f:
            regressor = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None

    # Load the preprocessed data for the specified company
    data_path = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed', f'{company}_historical_stock_prices_preprocessed.csv')
    data = pd.read_csv(data_path)

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
    print(f"Next week's {company} stock price predictions: {next_week_predictions.flatten()}")
    return next_week_predictions

make_predictions('AAPL')
make_predictions('MSFT')
make_predictions('GOOGL')    
