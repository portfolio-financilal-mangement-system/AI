def fetch_financial_news():
    import pandas as pd
    from newsapi import NewsApiClient

    try:
        # Initialize NewsApiClient with your API key
        api_key = 'bd7d2cd018d34e13a458621834f978c0'
        newsapi = NewsApiClient(api_key=api_key)

        # Define keywords and sources for news articles
        keywords = {
            'apple': ['apple stock price news'],
            'microsoft': ['Microsoft stock price news'],
            'google': ['Google stock price news'],
            'forex_egypt': ['forex egypt', 'Egyptian pound exchange rate', 'forex market Egypt'],
        }

        # Store dataframes for each company
        dataframes = {}

        # Fetch news articles based on keywords and sources for each company
        for company, keyword_list in keywords.items():
            articles = []
            for keyword in keyword_list:
                news = newsapi.get_everything(q=keyword, language='en', page_size=100)
                articles.extend(news['articles'])

            # Extract relevant data from news articles
            data = {
                'date': pd.to_datetime([article['publishedAt'] for article in articles]),  # Convert to Timestamp
                'content': [article['content'] for article in articles],
                'headlines': [article['title'] for article in articles]
            }

            # Create a DataFrame from the extracted data
            df = pd.DataFrame(data)

            # Calculate last week's and last month's dates
            today = pd.Timestamp('today', tz='UTC')
            last_week_start = today - pd.Timedelta(days=7)
            last_month_start = today - pd.DateOffset(months=1)

            # Filter data for last week and last month
            last_week_data = df[(df['date'] >= last_week_start) & (df['date'] < today)]
            last_month_data = df[(df['date'] >= last_month_start) & (df['date'] < today)]

            # Store the DataFrame in the dictionary as a tuple containing last week's and last month's data
            dataframes[company] = (last_week_data, last_month_data)

        # Return the dictionary containing dataframes for each company
        return dataframes

    except Exception as e:
        print("Error occurred:", str(e))
        return None
