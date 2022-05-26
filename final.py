import enum
from httpx import get

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

    print("--- Execution time: %s seconds ---" %
          (time.time() - start_time))

    start_time = time.time()
    cleaned = tweet_cleaner(data)
    print("--- Execution time: %s seconds ---" %
          (time.time() - start_time))

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
    print("--- Execution time: %s seconds ---" %
          (time.time() - start_time))
    test_data = grammed[:5000]
    grammed = grammed[5000:]

    start_time = time.time()
    (bag, unique, docs) = bow(grammed)
    print("--- Execution time: %s seconds ---" %
          (time.time() - start_time))

    start_time = time.time()

    matrix = tf_idf(docs, bag)
    print("--- Execution time: %s seconds ---" %
          (time.time() - start_time))

    def topic_coherence(SOM_matrix, lattice_used):
        word2id = Dictionary(grammed)
        topics = []
        topic_words = get_topic_words(SOM_matrix, unique, lattice_used)
        print(topic_words[0].keys())
        for topic in topic_words:
            topics.append([word[0] for word in topic.keys()])

        cm = CoherenceModel(topics=topics,
                            texts=grammed,
                            coherence='u_mass',
                            dictionary=word2id)
        coherence_score = cm.get_coherence()

        return coherence_score

    def get_coherence_scores():
        optimal_size = get_coherent_lattice()

        def get_coherent_lattice():
            lattice_numbers = []
            coherence_list = []
            for n in range(3, 7):
                lattice_size = (n, n)
                SOM_matrix = SOM(matrix, .3, lattice_size, 9000)
                coherence_list.append(topic_coherence(
                    SOM_matrix, lattice_size))

                lattice_numbers.append((n, n))
            return lattice_numbers[coherence_list.index(
                max(coherence_list))]
        optimal_iteration = get_coherent_iteration(optimal_size)

        def get_coherent_iteration(best_size):
            iterations_list = []
            coherence_list = []
            lattice_used = (best_size, best_size)
            for n in range(5000, 10001, 1000):
                SOM_matrix = SOM(matrix, .3, lattice_used, n)
                coherence_list.append(topic_coherence(
                    SOM_matrix, lattice_used))

                iterations_list.append(n)
            return iterations_list[coherence_list.index(
                max(coherence_list))]

        return optimal_size, optimal_iteration
    (lattice, iterations) = get_coherence_scores()
