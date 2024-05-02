import pandas as pd

df = pd.read_csv('/home/mohamed/Downloads/pandas/USD_to_EGP_exchange_rate.csv')


#a new Data Frame with no empty cells

new_df = df.dropna()



#Remove all rows with NULL values

df.dropna(inplace = True)

#Convert Into a Correct Format

df['Date'] = pd.to_datetime(df['Date'])


#Remove rows with a NULL value in the "Date" column:

df.dropna(subset=['Date'], inplace = True)


#Remove all duplicates:

df.drop_duplicates(inplace = True)

# Save the cleaned DataFrame to a new CSV file
new_df.to_csv('/home/mohamed/Downloads/pandas/cleaned_USD_TO_EGP_historical_stock_prices.csv', index=False)

# Print the final cleaned DataFrame
print(new_df.to_string())