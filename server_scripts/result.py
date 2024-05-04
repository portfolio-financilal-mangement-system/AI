import pandas as pd
import sys
import os

# Insert the parent directory of the current script into the Python path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, parent_dir)  # Go up one level to the parent directory

from data.raw.historical_Data.historical_exchange_rates import fetch_usd_to_egp_exchange_rate_and_save_to_csv
from data.raw.historical_Data.historical_stock_prices import fetch_stock_data_and_save_to_csv

from data.processed.processed_data import preprocess_csv_files

from src.USD_to_EGP_make_predictions import make_usd_to_egp_predictions
from src.MSFT_make_predictions import make_msft_predictions
from src.GOOGL_make_predictions import make_google_predictions
from src.AAPL_make_predictions import make_apple_predictions


def result():
    fetch_usd_to_egp_exchange_rate_and_save_to_csv()
    fetch_stock_data_and_save_to_csv()
    preprocess_csv_files()
    
    # Generate predictions
    usd_to_egp_predictions = make_usd_to_egp_predictions()
    apple_predictions = make_apple_predictions()
    google_predictions = make_google_predictions()
    msft_predictions = make_msft_predictions()

    return {
        "stock_predictions": {
            "Apple": {
                "predicted_prices": apple_predictions.flatten().tolist()
            },
            "Google": {
                "predicted_prices": google_predictions.flatten().tolist()
            },
            "Microsoft": {
                "predicted_prices": msft_predictions.flatten().tolist()
            }
        },
        "usd_to_egp_predictions": {
            "predicted_exchange_rates": usd_to_egp_predictions.flatten().tolist()
        },
        "news_analysis": {
            "Apple": {
                "positive": 3,
                "negative": 2,
                "neutral": 1
            },
            "Google": {
                "positive": 2,
                "negative": 1,
                "neutral": 2
            },
            "Microsoft": {
                "positive": 3,
                "negative": 1,
                "neutral": 1
            }
        }
    }
