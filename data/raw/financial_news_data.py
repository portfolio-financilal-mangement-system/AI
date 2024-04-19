import pandas as pd
from newsapi import NewsApiClient

# Initialize NewsApiClient with your API key
api_key = 'bd7d2cd018d34e13a458621834f978c0'
newsapi = NewsApiClient(api_key=api_key)

try:
    # Define keywords and sources for news articles
    keywords = ['apple stock price news', 'Microsoft stock price news', 'Google stock price news', 'finance Egypt', 'economic Egypt', 'business Egypt']
    sources = 'bbc-news,the-verge,aj-news,financial-times,news24,news-com,the-telegraph,alahram,al-jazeera-english,esquire,google-news,associated-press,al-jazeera-english,esquire,google-news,associated-press,al-jazeera-english,esquire,google-news,associated-press'

    # Fetch news articles based on keywords and sources
    articles = []
    for keyword in keywords:
        news = newsapi.get_everything(q=keyword, sources=sources, language='en', page_size=100)
        articles.extend(news['articles'])

    # Extract relevant data from news articles
    data = {
        'date': [article['publishedAt'] for article in articles],
        'content': [article['content'] for article in articles],
        'headlines': [article['title'] for article in articles]
    }

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file in the raw data folder
    df.to_csv('../data/raw/financial_news_data.csv', index=False)
    print("Financial news data saved successfully.")

except Exception as e:
    print("Error occurred:", str(e))
