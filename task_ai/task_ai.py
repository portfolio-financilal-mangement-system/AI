import pandas as pd
from sklearn.model_selection import train_test_split

# Read the entire dataset
df = pd.read_csv('all-data.csv', encoding='ISO-8859-1', header=None, names=['sentiment', 'text'])

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the train and test sets to separate CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
import matplotlib.pyplot as plt

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read CSV file for training data with specified encoding
train_df = pd.read_csv('train_data.csv', encoding='ISO-8859-1', header=None, names=['sentiment', 'text'])

# Read CSV file for testing data with specified encoding
test_df = pd.read_csv('test_data.csv', encoding='ISO-8859-1', header=None, names=['sentiment', 'text'])

# Text Preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    else:
        return ''

# Apply preprocessing to text data
train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
test_df['cleaned_text'] = test_df['text'].apply(preprocess_text)

# Tokenization and Lemmatization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(lemmatized_tokens)

# Apply tokenization and lemmatization
train_df['processed_text'] = train_df['cleaned_text'].apply(tokenize_and_lemmatize)
test_df['processed_text'] = test_df['cleaned_text'].apply(tokenize_and_lemmatize)

# Splitting the data
X_train, y_train = train_df['processed_text'], train_df['sentiment']
X_test, y_test = test_df['processed_text'], test_df['sentiment']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred_lr = lr_model.predict(X_test_tfidf)
f1_score_lr = f1_score(y_test, y_pred_lr, average='weighted')
print("F1 Score (Logistic Regression):", f1_score_lr)

# Print segment text along with sentiment analysis results
# print("Segment Text\t\tActual\t\tPredicted")
# for segment, actual, predicted in zip(X_test, y_test, y_pred_lr):
#     if actual == predicted:
#         print(f"\033[92m{segment[:50] + '...' if len(segment) > 50 else segment}\033[0m", "\t\t", actual, "\t\t", predicted)
#     else:
#         print(segment[:50] + '...' if len(segment) > 50 else segment, "\t\t", actual, "\t\t", predicted)

# Save the model and vectorizer
joblib.dump(lr_model, 'sentiment_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_lr)
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1 score for each class
precision = precision_score(y_test, y_pred_lr, average=None, labels=['negative', 'neutral', 'positive'])
recall = recall_score(y_test, y_pred_lr, average=None, labels=['negative', 'neutral', 'positive'])
f1 = f1_score(y_test, y_pred_lr, average=None, labels=['negative', 'neutral', 'positive'])

# Calculate weighted average of precision, recall, and F1 score
weighted_precision = precision_score(y_test, y_pred_lr, average='weighted')
weighted_recall = recall_score(y_test, y_pred_lr, average='weighted')
weighted_f1 = f1_score(y_test, y_pred_lr, average='weighted')

# Print precision, recall, and F1 score for each class
print("Precision:")
for i, label in enumerate(['negative', 'neutral', 'positive']):
    print(f"{label.capitalize()}: {precision[i]}")

print("\nRecall:")
for i, label in enumerate(['negative', 'neutral', 'positive']):
    print(f"{label.capitalize()}: {recall[i]}")

print("\nF1 Score:")
for i, label in enumerate(['negative', 'neutral', 'positive']):
    print(f"{label.capitalize()}: {f1[i]}")

# Print weighted average of precision, recall, and F1 score
print("\nWeighted Average:")
print("Precision:", weighted_precision)
print("Recall:", weighted_recall)
print("F1 Score:", weighted_f1)

# Generate classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
import pandas as pd

# Load the dataset
df = pd.read_csv('microsoft-news_data.csv')  # Replace 'microsoft-news_data.csv' with the path to your CSV file

# Extract the 'content' column as an array
content_array = df['headlines'].values

# Display the array
# print(content_array)

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

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
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Load the trained model and TF-IDF vectorizer
lr_model = joblib.load('sentiment_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')



# Preprocess the text segments
cleaned_segments = [clean_text(segment) for segment in content_array]
processed_segments = [remove_stopwords(segment) for segment in cleaned_segments]
lemmatized_segments = [lemmatize_text(segment) for segment in processed_segments]

# Vectorize the preprocessed text segments using the TF-IDF vectorizer
text_tfidf = tfidf_vectorizer.transform(lemmatized_segments)

# Predict the sentiment for each vectorized text segment
segment_sentiments = lr_model.predict(text_tfidf)

# Display the results
print("Segment Text\t\tPredicted Sentiment")
for segment, predicted_sentiment in zip(content_array, segment_sentiments):
    print(segment, "\t\t", predicted_sentiment)
# Count variables for positive, negative, and neutral sentiments
positive_count = 0
negative_count = 0
neutral_count = 0

# Iterate through the predicted sentiments and count occurrences
for sentiment in segment_sentiments:
    if sentiment == 'positive':
        positive_count += 1
    elif sentiment == 'negative':
        negative_count += 1
    else:  # Neutral sentiment
        neutral_count += 1

# Print the counts
print("Positive count:", positive_count)
print("Negative count:", negative_count)
print("Neutral count:", neutral_count)
