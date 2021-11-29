from modules.services import DatabaseConnection
from modules.gram import tweet_grammer, tweet_cleaner
from modules.vectorizer import bow, prune_bow, tf_idf
import time
from modules.tweet_preprocessor import count_english, remove_noneng_stopwords
import pandas as pd
from pprint import pprint as print

# conn = DatabaseConnection('mongodb://localhost:27017')


# start_time = time.time()
# raw = conn.get_raw_tweets()
# data = [tweet['full_text'] for tweet in raw]
# print("--- Execution time: %s seconds ---" % (time.time() - start_time))


start_time = time.time()
csv = pd.read_csv('./data/tweets_processed.csv')
data = csv['Content'].values[:13000]
print("--- Execution time: %s seconds ---" % (time.time() - start_time))


start_time = time.time()
cleaned = tweet_cleaner(data)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))


start_time = time.time()
grammed = tweet_grammer(cleaned)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))


start_time = time.time()
(bag, unique, docs) = bow(grammed)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))

start_time = time.time()

matrix = tf_idf(docs, bag)
print(matrix.shape)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))




# todo iterate over all docs and remove keywords not in unique
# todo remove empty docs

# (grams, unique, docs) = prune_bow(bag)


# print(len(unique))


