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
from data.raw.news.financial_news_data import save_news_data

from src.MSFT_stock_prediction import MSFT_predictions
from src.GOOGL_stock_prediction import GOOGL_predictions
from src.AAPL_stock_prediction import AAPL_predictions
from src.USD_to_EGP_stock_prediction import USD_to_EGP_predictions

# for news analysis
from src.news_analysis.sentiment_counter import count_sentiments
from src.news_analysis.sentiment_analysis import makeProcess_csv

def result():
    fetch_usd_to_egp_exchange_rate_and_save_to_csv()
    fetch_stock_data_and_save_to_csv()
    preprocess_csv_files()
    save_news_data()
    makeProcess_csv()

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
        "news_analysis_last_week":{
            "Apple": [count_sentiments('../data/interim/apple-news_sentiment.csv')],
            "Google": [count_sentiments('../data/interim/google-news_sentiment.csv')],
            "Microsoft": [count_sentiments('../data/interim/microsoft-news_sentiment.csv')],
            "forex": [count_sentiments('../data/interim/forex_egypt-news_sentiment.csv')],
        } 
    }


print("Result function executed successfully",result())