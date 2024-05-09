import os
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')

import os
import sys

# Get the current working directory
current_dir = os.path.dirname(__file__)

# Calculate the path to the project root directory by going up two directories
project_root = os.path.abspath(os.path.join(current_dir, '..', '..',))

# Add the project root directory to the Python path
sys.path.append(project_root)

# Now you can import the module
from data.raw.news.financial_news_data import save_news_data

# to update the news data
save_news_data()

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text)

    # Determine sentiment label based on compound score
    if sentiment_scores['compound'] > 0.05:
        return "Positive"
    elif sentiment_scores['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function to process each CSV file
def process_csv(input_filename, output_filename):
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return
    
    with open(input_filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    for item in data:
        # Analyze sentiment for the content
        item['predicted_sentiment'] = analyze_sentiment(item['content'])

    # Write the results to a new CSV file
    fieldnames = ['date', 'content', 'predicted_sentiment']
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow({key: item[key] for key in fieldnames})

    print(f"Sentiment analysis results saved to {output_filename}")

# Update file paths
input_files = {
    'apple_news': '../../data/raw/news/apple-news_data.csv',
    'microsoft_news': '../../data/raw/news/microsoft-news_data.csv',
    'google_news': '../../data/raw/news/google-news_data.csv',
    'forex_news': '../../data/raw/news/forex_egypt-news_data.csv'
}

output_files = {
    'apple_news': '../../data/interim/apple-news_sentiment.csv',
    'microsoft_news': '../../data/interim/microsoft-news_sentiment.csv',
    'google_news': '../../data/interim/google-news_sentiment.csv',
    'forex_news': '../../data/interim/forex_egypt-news_sentiment.csv'
}

# Process each CSV file
for key in input_files:
    process_csv(input_files[key], output_files[key])
