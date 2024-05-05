import os
import yfinance as yf
from datetime import datetime

def fetch_usd_to_egp_exchange_rate_and_save_to_csv():
    # Fetch historical USD to EGP exchange rate data up to the current date and time
    usd_to_egp_data = yf.download('EGP=X', start='2020-01-01', end=datetime.now())  

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Save the data to a CSV file in the same directory as the function
    csv_path = os.path.join(script_dir, 'USD_to_EGP_exchange_rate.csv')
    usd_to_egp_data.to_csv(csv_path)

# Call the function to fetch the data and save it to a CSV file
fetch_usd_to_egp_exchange_rate_and_save_to_csv()
