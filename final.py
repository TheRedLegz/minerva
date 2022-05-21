import enum

from sklearn import cluster
from modules.services import DatabaseConnection
from modules.gram import tweet_grammer, tweet_cleaner, tweet_pos
from modules.som import SOM, get_topic_words, tweet_find_cluster
from modules.vectorizer import bow, tf_idf
import numpy as np
import pandas as pd
import time
from pprint import pprint as print
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    conn = DatabaseConnection('mongodb://localhost:27017')

    start_time = time.time()
    raw = conn.get_raw_tweets()

    data = [tweet['full_text'] for tweet in raw]
    csv = pd.read_csv('./data/tweets_processed.csv')

    for tweet in csv['Content'].values[:30000]:
        data.append(tweet)

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

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
    test_data = grammed[:5000]
    grammed = grammed[5000:]

    start_time = time.time()
    (bag, unique, docs) = bow(grammed)
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()

    matrix = tf_idf(docs, bag)
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    lattice_size = (4, 4)
    (row, col) = lattice_size
    print(matrix.shape)
    SOM_matrix = SOM(matrix, .3, lattice_size, 4000)

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

    def topic_coherence():
        word2id = Dictionary(grammed)

        topic_words = get_topic_words(SOM_matrix, unique, lattice_size)
        topics = [topic['words'] for topic in topic_words]

        cm = CoherenceModel(topics=topics,
                            texts=grammed,
                            coherence='c_v',
                            dictionary=word2id)
        coherence_per_topic = cm.get_coherence_per_topic()

        topics_str = ['\n '.join(t) for t in topics]
        data_topic_score = pd.DataFrame(
            data=zip(topics_str, coherence_per_topic), columns=['Topic', 'Coherence'])
        data_topic_score = data_topic_score.set_index('Topic')

        fig, ax = plt.subplots(figsize=(2, 6))
        ax.set_title("Topics coherence\n $C_v$")
        sns.heatmap(data=data_topic_score, annot=True, square=True,
                    cmap='Reds', fmt='.2f',
                    linecolor='black', ax=ax)
        plt.yticks(rotation=0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.show(block=False)
        plt.pause(2000)

    # TODO: Print out keyword frequencies per topic
    for i in range(row):
        for j in range(col):
            print("[%d][%d]:" % (i, j))
            print(len(cluster_matrix[i][j]))
            
            # for tweet in cluster_matrix[i][j]:
            #     print(tweet)

    # todo iterate over all docs and remove keywords not in unique
    # todo remove empty docs

    # (grams, unique, docs) = prune_bow(bag)
    # print(len(unique))
