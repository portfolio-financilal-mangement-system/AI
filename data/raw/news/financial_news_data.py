import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import re
import os

# Initialize NewsApiClient with your API key
api_key = 'bd7d2cd018d34e13a458621834f978c0'
newsapi = NewsApiClient(api_key=api_key)

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_news_data():
    try:
        # Get the current year
        current_year = datetime.now().year

        # Define keywords and sources for news articles
        keywords = {
            'apple': ['apple stock price news'],
            'microsoft': ['Microsoft stock price news'],
            'google': ['Google stock price news'],
            'forex_egypt': ['forex Egypt', 'Egyptian pound exchange rate', 'forex market Egypt', 'Egypt economy', 'dollar exchange rate in Egypt', 'currency conversion in Egypt',
                    'Egyptian economy news', 'Egyptian currency updates', 'Egyptian financial market', 'Cairo stock exchange', 'Egyptian GDP growth', 'Egyptian inflation rate',
                    'Egyptian monetary policy', 'Central Bank of Egypt'],
        }

        # Get the current working directory
        current_directory = os.getcwd()

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
                'content': [clean_text(article['content']) for article in articles],  # Clean the content
                'headlines': [article['title'] for article in articles]
            }

            # Filter articles that are from the current year
            data_filtered = {
                'date': [],
                'content': [],
                'headlines': []
            }
            for i in range(len(data['date'])):
                article_year = datetime.strptime(data['date'][i], '%Y-%m-%dT%H:%M:%SZ').year
                if article_year == current_year:
                    data_filtered['date'].append(data['date'][i])
                    data_filtered['content'].append(data['content'][i])
                    data_filtered['headlines'].append(data['headlines'][i])

            # Create a DataFrame from the filtered data
            df = pd.DataFrame(data_filtered)

            # Convert 'date' column to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Remove duplicate rows based on content
            df.drop_duplicates(subset=['content'], inplace=True)

            # Sort the DataFrame by date
            df = df.sort_values(by='date')

            # Define the filename based on the company
            filename = f'{company}-news_data.csv'

            # Construct the file path to save the CSV file in the function's directory
            file_path = os.path.join(current_directory, filename)

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)

            print(f"{company.capitalize()} news data saved successfully in {file_path}.")

    except Exception as e:
        print("Error occurred:", str(e))

# Call the function to save the news data
save_news_data()
