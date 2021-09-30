from modules.gram import gram_documents
from pprint import pprint as print
import pandas as pd



# import test data

data = pd.read_csv('data/test_tweets.csv')
tweets = data['Tweet']

# preprocess data

tokens = gram_documents(tweets)


# vectorize



# som



print(tokens[:3])