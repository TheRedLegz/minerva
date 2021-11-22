from modules.gram import gram_documents
from pprint import pprint as print

from pymongo import MongoClient

import numpy as np
import json
from requests.api import get
from modules.tweet_preprocessor import preprocess_documents, preprocess_tweet, spell_check
from modules.word2vec import get_word2vec_from_data
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
from modules.pca import pca
from modules.lsi import lsi
from matplotlib import pyplot as plt
from modules.som import SOM, find_topics, print_data_to_SOM
from modules.sentiment import sentimentinator
from nltk.corpus import stopwords
from modules.services import DatabaseConnection

import pandas as pd

# client = MongoClient('mongodb://localhost:27017')
# db_raw = client['minerva_raw_tweets']
# rawtweets = db_raw['rawtweets']


if __name__ == "__main__":
    db = DatabaseConnection('mongodb://localhost:27017')
    unprocessed_tweets = db.get_unprocessed_tweets()

    # From CSV
    # data = pd.read_csv('covid19_tweets.csv')
    # data = data['text'].values

    print("Starting Preprocessing")
    print("Unprocessed Tweet count: " + str(len(unprocessed_tweets)))
    preprocessed_tweets = preprocess_documents(unprocessed_tweets, 20)
    db.add_to_collection(preprocessed_tweets, 'preprocessed_tweets')

    # TF-IDF implementation
    print("Starting BOW")
    preprocessed_tweets = db.get_preprocessed_tweets()
    bowres = bag_of_words(preprocessed_tweets, to_preprocess=False)
    (bow, unique, doc_grams) = bowres

    print("Starting Pruning")
    (bow, unique, doc_grams) = prune_bow(bowres, 5)

    # print("Starting TF-IDF")
    # vectors = tf_idf(preprocessed_tweets, bow)

    # # PCA (to be decided)
    # # vectors_t = np.transpose(vectors)
    # # (lsi_matrix, sum) = pca(vectors_t)

    # lattice_size = (4, 4)
    # (row, col) = lattice_size

    # # TF-IDF implementation
    # # SOM_matrix = SOM(lsi_matrix, .5, lattice_size)
    # SOM_matrix = SOM(vectors, .5, lattice_size)

    # print("\nFinal SOM weights")
    # print("Lattice size: (%d, %d)" % (row, col))

    # print("\nThe Clustered Topics")
    # # print_data_to_SOM(SOM_matrix, lsi_matrix, unique)
    # # print_data_to_SOM(SOM_matrix, vectors, original_data)
    # sentimentinator(data)

    # data_selected_index = 0
    # while(data_selected_index != -1):
    #     print("\nSelect a tweet ( 0 -", len(data)-1, "): ")
    #     data_selected_index = int(input())

    #     if data_selected_index != -1:
    #         topics = find_topics(SOM_matrix, vectorized_words, doc_grams[data_selected_index], unique, lattice_size)

    #         print("\nRaw Tweet:\n", db_results[data_selected_index]['data']['full_text'])
    #         print("\nSelected Doc:", doc_grams[data_selected_index])
    #         print("\nTopics:")
    #         for topic in topics:
    #             (location, word) = topic
    #             (x, y) = location
    #             print("[", x, "][", y,"] =", word)


    # TO BE DISCARDED
    # WORD2VEC implementation
    # model = get_word2vec_from_data(data, to_preprocess=False)

    # vectorized_words = []
    # unique = []
    # for i in range(len(model.wv.index_to_key)):
    #     word = model.wv.index_to_key[i]
    #     unique.append(word)
    #     vectorized_words.append(model.wv[word])
    # vectorized_words = np.asarray(vectorized_words)

    # doc_grams = []
    # for sentence in data:
    #     doc_grams.append([word for word in sentence if word in unique])

    # WORD2VEC implementation
    # SOM_matrix = SOM(vectorized_words, .5, lattice_size)
    print("---PROGRAM EXITED---")
