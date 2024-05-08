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

from src.MSFT_stock_prediction import MSFT_predictions
from src.GOOGL_stock_prediction import GOOGL_predictions
from src.AAPL_stock_prediction import AAPL_predictions
from src.USD_to_EGP_stock_prediction import USD_to_EGP_predictions


def result():
    fetch_usd_to_egp_exchange_rate_and_save_to_csv()
    fetch_stock_data_and_save_to_csv()
    preprocess_csv_files()
    
    return {
        "stock_predictions": {
            "Apple": {
                "predicted_prices": AAPL_predictions()
            },
            "Google": {
                "predicted_prices": GOOGL_predictions()
            },
            "Microsoft": {
                "predicted_prices": MSFT_predictions()
            }
        },
        "usd_to_egp_predictions": {
            "predicted_exchange_rates": USD_to_EGP_predictions()
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


print("Result function executed successfully",result())