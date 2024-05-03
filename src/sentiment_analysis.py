import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

# Function to predict sentiment for a given text
def predict_sentiment(text, tokenizer, model):
    encoded_text = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded_text)
        predictions = torch.nn.functional.softmax(output.logits, dim=-1)
        sentiment = torch.argmax(predictions).item()
        sentiment_labels = ["Positive", "Negative", "Neutral"]  # Modify based on your model

        # Return a dictionary with sentiment class and score
        return {
            "class": sentiment_labels[sentiment],
            "score": predictions.max().item()
        }

# Function to perform sentiment analysis for a given dataframe
def perform_sentiment_analysis(df, model_name):
    # Choose a pre-trained DistilBERT model for sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Get sentiment analysis results for each article in the dataframe
    sentiment_analysis = df["content"].apply(lambda x: predict_sentiment(x, tokenizer, model))

    # Get sentiment counts
    sentiment_counts = sentiment_analysis.apply(lambda x: x['class']).value_counts()

    # Return the sentiment analysis results and counts
    return {
        "sentiment_analysis": sentiment_analysis,
        "sentiment_counts": sentiment_counts
    }

# Main function
def main():
    # Dictionary to store sentiment counts for each company
    sentiment_counts_dict = {}

    # Define the model name
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    # Define the path to the data.raw directory
    data_raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    print(data_raw_dir)
    # Define the path to the news subdirectory
    path_dir = os.path.join(data_raw_dir, 'raw')
    news_dir = os.path.join(path_dir, 'news')
    print(news_dir)

    try:
        # Get list of CSV files in the news directory
        csv_files = [file for file in os.listdir(news_dir) if file.endswith("-news_data.csv")]

        # Iterate over each CSV file
        for csv_file in csv_files:
            # Load CSV file into a DataFrame
            df = pd.read_csv(os.path.join(news_dir, csv_file))

            # Convert 'date' column to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Extract company name from file name
            company = csv_file.split('-')[0]

            print(f"Sentiment analysis for {company.capitalize()}:")
            # Perform sentiment analysis for the dataframe
            results = perform_sentiment_analysis(df, model_name)
            sentiment_counts_dict[company] = results["sentiment_counts"]

    except FileNotFoundError:
        print("No CSV files found in the 'news' directory.")

    # Return sentiment counts dictionary
    return sentiment_counts_dict

if __name__ == "__main__":
    sentiment_counts = main()
    print(sentiment_counts)
