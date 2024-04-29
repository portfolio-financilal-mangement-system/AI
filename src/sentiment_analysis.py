import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

# Insert the parent directory of the current script into the Python path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))

# Assuming financial_news_data is a function that fetches financial news data
from data.raw.news.financial_news_data import fetch_financial_news

# Function to perform sentiment analysis for a given dataframe
def perform_sentiment_analysis(df, model_name):
    # Choose a pre-trained DistilBERT model for sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Get sentiment analysis results for each article in the dataframe
    sentiment_analysis = df["content"].apply(lambda x: predict_sentiment(x, tokenizer, model))
    
    # Return the sentiment analysis results
    return sentiment_analysis

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

# Main function
def main():
    # Load financial news data
    dataframes = fetch_financial_news()
    
    # Define the model name
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Iterate over each item in the dictionary
    for company, (last_week_data, last_month_data) in dataframes.items():
        print(f"Sentiment analysis for {company}:")

        # Perform sentiment analysis for last week
        last_week_sentiment_analysis = perform_sentiment_analysis(last_week_data, model_name)
        last_week_sentiment_counts = last_week_sentiment_analysis.apply(lambda x: x['class']).value_counts()
        print("Last Week Sentiment:")
        print("Positive:", last_week_sentiment_counts.get('Positive', 0))
        print("Negative:", last_week_sentiment_counts.get('Negative', 0))
        print("Neutral:", last_week_sentiment_counts.get('Neutral', 0))

        # Perform sentiment analysis for last month
        last_month_sentiment_analysis = perform_sentiment_analysis(last_month_data, model_name)
        last_month_sentiment_counts = last_month_sentiment_analysis.apply(lambda x: x['class']).value_counts()
        print("\nLast Month Sentiment:")
        print("Positive:", last_month_sentiment_counts.get('Positive', 0))
        print("Negative:", last_month_sentiment_counts.get('Negative', 0))
        print("Neutral:", last_month_sentiment_counts.get('Neutral', 0))
        
        print()

if __name__ == "__main__":
    main()
