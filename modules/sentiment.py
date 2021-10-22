import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentimentinator(dataSentiment):
    sentiRes = pd.DataFrame()
    analyser = SentimentIntensityAnalyzer()
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