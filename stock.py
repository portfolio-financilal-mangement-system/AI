import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score
from sklearn.impute import SimpleImputer

# Load your dataset (replace 'your_data.csv' with the actual file path)
data = pd.read_csv('AAPL_historical_stock_prices.csv')

# Impute missing value in the 'Change' column with the mean
imputer = SimpleImputer(strategy='mean')
data['Change'] = imputer.fit_transform(data[['Change']])

# Select relevant features and target
X = data[['Open', 'High', 'Low', 'Change']]
y = data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict stock prices for each day in the specified date range
date_range = pd.date_range(start='2024-05-20', end='2024-05-27', freq='D')
for date in date_range:
    future_date_features = pd.DataFrame({
        'Open': [148],
        'High': [150],
        'Low': [158],
        'Change': [2]
    })
    predicted_price = model.predict(future_date_features)
    print(f"Predicted stock price on {date.strftime('%Y-%m-%d')}: ${predicted_price[0]:.2f}")
