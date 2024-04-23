import pandas as pd

df = pd.read_csv('/home/mohamed/Downloads/pandas/USD_to_EGP_exchange_rate.csv')

print(df.to_string())

#a new Data Frame with no empty cells

new_df = df.dropna()

print(new_df.to_string())


#Remove all rows with NULL values

df.dropna(inplace = True)

#Convert Into a Correct Format

df['Date'] = pd.to_datetime(df['Date'])

print(df.to_string())

#Remove rows with a NULL value in the "Date" column:

df.dropna(subset=['Date'], inplace = True)

print(df.to_string())

#Remove all duplicates:

df.drop_duplicates(inplace = True)
