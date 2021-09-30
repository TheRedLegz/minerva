from pymongo import MongoClient
from pprint import pprint

import numpy as np
import json
from requests.api import get
from modules.preprocessor import preprocess_documents, write_to_file
from modules.word2vec import get_word2vec_from_data
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
from modules.pca import pca
from modules.lsi import lsi
from matplotlib import pyplot as plt
from modules.som import SOM, find_topics, print_data_to_SOM

import pandas as pd

client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']

if __name__ == "__main__":


    f = open('file2.json',encoding="utf8")
    db_results = json.load(f)
    f.close()

    # data = []
    data = pd.read_csv('test_tweets.csv')
    data = data['Tweet'].values
    # All tweets
    # for tweet in db_results:
    #     data.append(tweet['tweet'])

    # From MongoDB
    # db_results = list(rawtweets.find())
    # data = []
    # for a in db_results:s
    #     data.append(a['data']['full_text'])

    print("Starting Preprocessing")
    data = preprocess_documents(data[:50])
    print("Finished Preprocessing")

    #WORD2VEC implementation
    model = get_word2vec_from_data(data, to_preprocess=False)

    vectorized_words = []
    unique = []
    for i in range(len(model.wv.index_to_key)):
        word = model.wv.index_to_key[i]
        unique.append(word)
        vectorized_words.append(model.wv[word])
    vectorized_words = np.asarray(vectorized_words)
    print(vectorized_words.shape)

    doc_grams = []
    for sentence in data:
        doc_grams.append([word for word in sentence if word in unique])

    print(doc_grams)
    


    # TF-IDF implementation
    # bowres = bag_of_words(data, to_preprocess=False)
    # (bow, unique, doc_grams) = bowres

    # (bow, unique, doc_grams) = prune_bow(bowres, 3)

    # vectors = tf_idf(data, bow)

    # vectors_t = np.transpose(vectors)
    # (lsi_matrix, sum) = lsi(vectors_t)
    
    # print(lsi_matrix.shape )

    lattice_size = (2, 2)
    (row, col) = lattice_size

    #WORD2VEC implementation
    SOM_matrix = SOM(vectorized_words, .5, lattice_size)
    #TF-IDF implementation
    # SOM_matrix = SOM(lsi_matrix, .5, lattice_size)

    print("\nFinal SOM weights")
    print("Lattice size: (%d, %d)" %(row, col))

    for i in range(row):
        for j in range(col):
            print("[", i, "] [", j, "] =", SOM_matrix[i][j][:3])

    print("\nThe Clustered Topics")
    print_data_to_SOM(SOM_matrix, vectorized_words, unique)

    data_selected_index = 0
    while(data_selected_index != -1):
        print("\nSelect a tweet ( 0 -", len(data)-1, "): ")
        data_selected_index = int(input())

        if data_selected_index != -1:    
            topics = find_topics(SOM_matrix, vectorized_words, doc_grams[data_selected_index], unique, lattice_size)

            print("\nRaw Tweet:\n", db_results[data_selected_index]['data']['full_text'])
            print("\nSelected Doc:", doc_grams[data_selected_index])
            print("\nTopics:")
            for topic in topics:
                (location, word) = topic
                (x, y) = location
                print("[", x, "][", y,"] =", word)



    print("---PROGRAM EXITED---")
