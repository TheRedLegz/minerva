import enum
from modules.services import DatabaseConnection
from modules.gram import tweet_grammer, tweet_cleaner, tweet_pos
from modules.som import SOM, tweet_find_cluster
from modules.vectorizer import bow, prune_bow, _tf_idf_sub, tf_idf
import numpy as np
import pandas as pd
import time
from pprint import pprint as print

conn = DatabaseConnection('mongodb://localhost:27017')


start_time = time.time()
raw = conn.get_raw_tweets()

# data = [tweet['full_text'] for tweet in raw]

# data = data[:10000]
# csv = pd.read_csv('./data/tweets_processed.csv')

# for tweet in csv['Content'].values[:30000]:
#     data.append(tweet)

csv = pd.read_csv('./data/tweets_processed.csv')
data = [tweet for tweet in csv['Content'].values[:30000]]
for i in range(10000):
    data.append(raw[i]['full_text'])

print("--- Execution time: %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# csv = pd.read_csv('./data/tweets_processed.csv')
# data = csv['Content'].values[:13000]
# print("--- Execution time: %s seconds ---" % (time.time() - start_time))


start_time = time.time()
cleaned = tweet_cleaner(data)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))

start_time = time.time()
grammed = tweet_grammer(cleaned)
for i, tweet_grams in enumerate(grammed):
    grammed[i] = ' '.join(tweet_grams)
pos_tags = tweet_pos(grammed)
for i, tweet_grams in enumerate(pos_tags):
    temp = []
    for gram in tweet_grams:
        if gram[1].startswith('NN'):
            temp.append(gram[0])
    grammed[i] = temp
print("--- Execution time: %s seconds ---" % (time.time() - start_time))
test_data = grammed[:10000]
grammed = grammed[10000:]

start_time = time.time()
(bag, unique, docs) = bow(grammed)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))

start_time = time.time()

matrix = tf_idf(docs, bag)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))

lattice_size = (4, 4)
(row, col) = lattice_size

SOM_matrix = SOM(matrix, .3, lattice_size)

cluster_matrix = np.empty(shape=(row, col), dtype=object)

for i in range(row):
    for j in range(col):
        cluster_matrix[i][j] = []

# preprocessed_tweets = grammed

for i, tweet in enumerate(test_data):
    (result_matrix, bmu) = tweet_find_cluster(
        SOM_matrix, lattice_size, tweet, unique)
    (bmu_row, bmu_col) = bmu

    cluster_matrix[bmu_row][bmu_col].append(tweet)
    # cluster_matrix[bmu_row][bmu_col].append(data[i])


# TODO: Print out keyword frequencies per topic
for i in range(row):
    for j in range(col):
        print("[%d][%d]:" % (i, j))
        print(len(cluster_matrix[i][j]))


#             print(tweet)


# todo iterate over all docs and remove keywords not in unique
# todo remove empty docs

# (grams, unique, docs) = prune_bow(bag)
# print(len(unique))
