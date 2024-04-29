def fetch_financial_news():
    import pandas as pd
    from newsapi import NewsApiClient

    # Initialize NewsApiClient with your API key
    api_key = 'bd7d2cd018d34e13a458621834f978c0'
    newsapi = NewsApiClient(api_key=api_key)

    try:
        # Define keywords and sources for news articles
        keywords = {
            'apple': ['apple stock price news'],
            'microsoft': ['Microsoft stock price news'],
            'google': ['Google stock price news'],
            'egypt' : ['Egypt US exchange rate'],
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
                'date': [article['publishedAt'] for article in articles],
                'content': [article['content'] for article in articles],
                'headlines': [article['title'] for article in articles]
            }

            # Create a DataFrame from the extracted data
            df = pd.DataFrame(data)

            # # Define the filename based on the company
            # filename = f'{company}-news_data.csv'

            # # Save the DataFrame to a CSV file
            # df.to_csv(filename, index=False)

            # Store the DataFrame in the dictionary
            dataframes[company] = df

        # Return the dictionary containing dataframes for each company
        return dataframes

    except Exception as e:
        print("Error occurred:", str(e))

# Call the function
getDataframes = fetch_financial_news()
