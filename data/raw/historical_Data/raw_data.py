 import pandas as pd

 def load_stock_data(ticker):
     filename = f"{ticker}_historical_stock_prices.csv"
     return pd.read_csv(filename)

 def load_exchange_rate_data():
     return pd.read_csv("USD_to_EGP_exchange_rate.csv")
