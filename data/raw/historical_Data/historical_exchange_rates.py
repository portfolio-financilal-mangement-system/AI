import yfinance as yf
from datetime import datetime

# Fetch historical USD to EGP exchange rate data up to the current date and time
usd_to_egp_data = yf.download('EGP=X', start='2020-01-01', end=datetime.now())  

# Calculate the percentage change in exchange rate
usd_to_egp_data['Change (%)'] = (usd_to_egp_data['Close'].diff() / usd_to_egp_data['Close'].shift(1)) * 100

# Save the data to a CSV file
usd_to_egp_data.to_csv('USD_to_EGP_exchange_rate.csv')

