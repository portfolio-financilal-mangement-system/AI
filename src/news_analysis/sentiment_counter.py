import csv
import os

def count_sentiments(input_filename):
    # Dictionary to store sentiment counts
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

    # Check if input file exists
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return sentiment_counts

    # Open CSV file and count sentiments
    with open(input_filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentiment = row['predicted_sentiment']
            if sentiment == 'Neutral':
                sentiment_counts['neutral'] += 1
            elif sentiment == 'Positive':
                sentiment_counts['positive'] += 1
            elif sentiment == 'Negative':
                sentiment_counts['negative'] += 1
    # print(sentiment_counts)
    return sentiment_counts

print("apple",count_sentiments('../../data/interim/apple-news_sentiment.csv'))
print("google",count_sentiments('../../data/interim/google-news_sentiment.csv'))
print("msft",count_sentiments('../../data/interim/microsoft-news_sentiment.csv'))
print("fx",count_sentiments('../../data/interim/forex_egypt-news_sentiment.csv'))
