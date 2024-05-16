import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Text Preprocessing functions
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text
    else:
        return ''

def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()  # Define lemmatizer here
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Load the trained model and TF-IDF vectorizer
lr_model = joblib.load('sentiment_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the dataset
df = pd.read_csv('google-news_data.csv')

# Extract the 'content' column as an array
content_array = df['headlines'].values

# Preprocess the text segments
cleaned_segments = [clean_text(segment) for segment in content_array]
processed_segments = [remove_stopwords(segment) for segment in cleaned_segments]
lemmatized_segments = [lemmatize_text(segment) for segment in processed_segments]

# Vectorize the preprocessed text segments using the TF-IDF vectorizer
text_tfidf = tfidf_vectorizer.transform(lemmatized_segments)

# Predict the sentiment for each vectorized text segment
segment_sentiments = lr_model.predict(text_tfidf)

# Count the occurrences of positive, negative, and neutral sentiments
sentiment_counts = pd.Series(segment_sentiments).value_counts()
# Display the results
print("Segment Text\t\tPredicted Sentiment")
for segment, predicted_sentiment in zip(content_array, segment_sentiments):
    print(segment, "\t\t", predicted_sentiment)

# Print the count of each sentiment
print("Sentiment Counts:")
print(sentiment_counts)

# Plotting the distribution of sentiments
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
