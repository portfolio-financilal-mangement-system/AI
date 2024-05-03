import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import re

# Initialize NewsApiClient with your API key
api_key = 'bd7d2cd018d34e13a458621834f978c0'
newsapi = NewsApiClient(api_key=api_key)

try:
    # Define keywords and sources for news articles
    keywords = {
        'apple': ['apple stock price news'],
        'microsoft': ['Microsoft stock price news'],
        'google': ['Google stock price news'],
        'forex_egypt': ['forex egypt', 'Egyptian pound exchange rate', 'forex market Egypt'],
    }
    
    # Calculate the date range for the last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    # Fetch news articles based on keywords and sources for each company
    for company, keyword_list in keywords.items():
        articles = []
        for keyword in keyword_list:
            news = newsapi.get_everything(q=keyword, language='en', page_size=100, from_param=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))
            articles.extend(news['articles'])

        # Extract relevant data from news articles
        data = {
            'date': [article['publishedAt'] for article in articles],
            'content': [re.sub(r'<.*?>', '', article['content']) if article['content'] else '' for article in articles],  # Remove all HTML elements
            'headlines': [article['title'] for article in articles]
        }

        # Create a DataFrame from the extracted data
        df = pd.DataFrame(data)

        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort the DataFrame by date
        df = df.sort_values(by='date')

        # Define the filename based on the company
        filename = f'{company}-news_data.csv'

        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)

        print(f"{company.capitalize()} news data saved successfully.")

except Exception as e:
    print("Error occurred:", str(e))
