import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load your custom CSV dataset
dataset = pd.read_csv('Tweets.csv')

# Define a function to determine sentiment using VADER
def get_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Define a function to determine sentiment using TextBlob
def get_sentiment_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to each row in the dataset
dataset['sentiment_vader'] = dataset['text'].apply(get_sentiment_vader)
dataset['sentiment_textblob'] = dataset['text'].apply(get_sentiment_textblob)

# Display the results
print(dataset[['text', 'sentiment_vader', 'sentiment_textblob']])
