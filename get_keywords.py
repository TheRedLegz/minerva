"""

NOT PART OF THE PROCESS
THIS IS JUST TO FIND THE RIGHT KEYWORDS FOR OUR SCRAPER


todo

scrape tweets using a bunch of keywords /
clean the tweets / 
get the unigrams /

get the bigrams
- simple clean for test tweets / 
- train a bigram model using the downloaded tweets about education /
- save it
- simple clean for own tweets /
- extract the bigrams from our own tweets /
- remove bigrams having a stopword /
- lemmatize the bigram / 
- put in a dict /

get the trigrams
- train a trigram model using the bigrams from our downloaded tweets about education /
- extract the trigrams from our own tweets /


"""


from pickle import STOP
from gensim.utils import tokenize
import pandas as pd
from pprint import pprint as print
import tweet_preprocessor as p
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary

import json
test_tweets = pd.read_csv('test_tweets.csv')
test_data = test_tweets['Tweet']

clean_test_data = p.clean_documents(test_data)
tokenized_data = [list(tokenize(doc, deacc=True, lower=True)) for doc in clean_test_data]

bigram_phrases = Phrases(tokenized_data, min_count=1, threshold=50)
trigram_phrases = Phrases(bigram_phrases[tokenized_data], min_count=3, threshold=10)




bigram_model = Phraser(bigram_phrases)
trigram_model = Phraser(trigram_phrases)

bigram_model
file = open('file2.json',encoding="utf8")
data = json.load(file)
data = [doc['tweet'] for doc in data]


tokenized_tweets = p.clean_documents(data)
tokenized_tweets = [list(tokenize(doc, deacc=True, lower=True)) for doc in tokenized_tweets]

tokens = trigram_model[bigram_model[tokenized_tweets]]
cleaned_tokens = []


def clean_tokens(doc):
    res = []
    for word in doc:
        if '_' in word:
            split = word.split('_')

            if len(split[0]) > 2 and len(split[1]) > 2 and split[1] not in STOPWORDS and split[0] not in STOPWORDS:
                res.append(p.lemmatize(split[0]) + '_' + p.lemmatize(split[1]))


        elif len(word) > 2 and word not in STOPWORDS:
            res.append(word)

    return res

cleaned_tokens = list(map(clean_tokens, tokens))


def get_token_count(dict):
    keys = dict.token2id
    cfs = dict.cfs

    result = {}
    
    for key in cfs:
        token_name = list(keys.keys())[list(keys.values()).index(key)]
        result[token_name] = cfs[key]

    return result

dictionary = Dictionary(cleaned_tokens)
testtt = dictionary.token2id.keys()


unique = []

for bigram in testtt:
    if bigram not in unique and '_' in bigram:
        unique.append(bigram)

print(unique)
print(sorted(get_token_count(dictionary).items(), key=lambda x: x[1], reverse=True)[:150])



