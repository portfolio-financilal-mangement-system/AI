import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

def fetch_stock_data_and_save_to_csv():
    # Define stock tickers for Apple, Microsoft, and Google
    stock_tickers = ["AAPL", "MSFT", "GOOGL"]

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Fetch historical stock prices for each company
    for ticker in stock_tickers:
        # Fetch historical data
        stock_data = yf.download(ticker, start='2020-01-01', end=datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        # Save historical data to CSV file in the same directory as the function
        csv_path = os.path.join(script_dir, f'{ticker}_historical_stock_prices.csv')
        stock_data.to_csv(csv_path)

    print("Historical data saved to CSV files.")

# Call the function to fetch the data and save it to CSV files
# fetch_stock_data_and_save_to_csv()
