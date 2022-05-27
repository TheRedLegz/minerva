import pandas as pd
import nltk
nltk.download('state_union')

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

train_text = state_union.raw("2005-GWBush.txt")
tokenizer = PunktSentenceTokenizer(train_text)

analyser = SentimentIntensityAnalyzer()


def sentimentinator(dataSentiment):
    sentiRes = pd.DataFrame()
    sentiment_score_list = []
    sentiment_label_list = []
    tweet_list = []
    for i in dataSentiment:
        sentiment_score = analyser.polarity_scores(i)

        if sentiment_score['compound'] >= 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Positive')
            tweet_list.append(i)
        elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Neutral')
            tweet_list.append(i)
        elif sentiment_score['compound'] <= -0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Negative')
            tweet_list.append(i)
    
    sentiRes['sentiment'] = sentiment_label_list
    sentiRes['sentiment_score'] = sentiment_score_list
    sentiRes['tweet'] = tweet_list

    return sentiRes

def get_sentiment(tweet):
    sentiment = "neutral"
    
    score = analyser.polarity_scores(tweet)['compound']
    if score <= -0.05:
        sentiment = "negative"
    elif score < 0.05:
        sentiment = "neutral"
    elif score > 0.05:
        sentiment = "positive"

    return (sentiment, score)

def chunker(string):
    return tokenizer.tokenize(string)
