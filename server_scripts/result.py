import pandas as pd
import sys
import os

# Insert the parent directory of the current script into the Python path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))  # Go up one level to the parent directory

# Now you should be able to import from the src directory
from src.sentiment_analysis import main as get_sentiment_counts

# Get the sentiment counts
sentiment_counts = get_sentiment_counts()

def result():
    # Convert the sentiment_counts dictionary to a JSON-serializable format
    serialized_counts = {}
    for company, counts in sentiment_counts.items():
        serialized_counts[company] = {
            "last_week": counts["last_week"].to_dict(),
            "last_month": counts["last_month"].to_dict()
        }
    return {
        "news analysis": serialized_counts,
    }
