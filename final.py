from modules.services import DatabaseConnection
from modules.gram import tweet_grammer, tweet_cleaner
from modules.som import SOM, tweet_find_cluster
from modules.vectorizer import bow, prune_bow, _tf_idf_sub, tf_idf
import numpy as np
import pandas as pd
import time
from pprint import pprint as print

conn = DatabaseConnection('mongodb://localhost:27017')


start_time = time.time()
raw = conn.get_raw_tweets()
data = [tweet['full_text'] for tweet in raw]
print("--- Execution time: %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# csv = pd.read_csv('./data/tweets_processed.csv')
# data = csv['Content'].values[:13000]
# print("--- Execution time: %s seconds ---" % (time.time() - start_time))


start_time = time.time()
cleaned = tweet_cleaner(data[:5000])
print("--- Execution time: %s seconds ---" % (time.time() - start_time))


start_time = time.time()
grammed = tweet_grammer(cleaned)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))


start_time = time.time()
(bag, unique, docs) = bow(grammed)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))

start_time = time.time()

matrix = tf_idf(docs, bag)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))

lattice_size = (4, 4)
(row, col) = lattice_size

SOM_matrix = SOM(matrix, .5, lattice_size)

cluster_matrix = np.empty(shape=(row, col), dtype=object)

for i in range(row):
    for j in range(col):
        cluster_matrix[i][j] = []

preprocessed_tweets = grammed

for tweet in preprocessed_tweets[1000:2000]:
    (result_matrix, bmu) = tweet_find_cluster(SOM_matrix, lattice_size, tweet, unique)
    (bmu_row, bmu_col) = bmu
    
    cluster_matrix[bmu_row][bmu_col].append(tweet)

for i in range(row):
    for j in range(col):
        print("[%d][%d]:" % (i, j))
        for tweet in cluster_matrix[i][j]:
            print(tweet)




# todo iterate over all docs and remove keywords not in unique
# todo remove empty docs

# (grams, unique, docs) = prune_bow(bag)
# print(len(unique))


