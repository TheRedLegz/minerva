from pymongo import MongoClient
from pprint import pprint

import numpy as np
import nltk
from modules.preprocessor import preprocess_documents, write_to_file
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
from modules.pca import pca
from modules.lsi import lsi
from matplotlib import pyplot as plt
from modules.som import SOM, find_topics, print_data_to_SOM
client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']
nltk.download("words")

if __name__ == "__main__":

    # Fetching scraped tweets from MongoDB
    db_results = list(rawtweets.find())
    data = []

    print("Started Loading Data")
    # Assigning tweets in variable

    for a in db_results:
        data.append(a['data']['full_text'])
    data = preprocess_documents(data)
    print(data)
    bowres = bag_of_words(data, to_preprocess=False)
    (bow, unique, doc_grams) = bowres

    print("Finished BOW\n")

    (bow, unique, doc_grams) = prune_bow(bowres, 3)

    vectors = tf_idf(data, bow)

    vectors_t = np.transpose(vectors)
    (lsi_matrix, sum) = lsi(vectors_t)
    
    print(lsi_matrix.shape )

    lattice_size = (2, 3)
    (row, col) = lattice_size

    SOM_matrix = SOM(lsi_matrix,.5, lattice_size)
    
    print("\nFinal SOM weights")
    print("Lattice size: (%d, %d)" %(row, col))

    for i in range(row):
        for j in range(col):
            print("[", i, "] [", j, "] =", SOM_matrix[i][j][:3])

    print("\nThe Clustered Topics")
    print_data_to_SOM(SOM_matrix, lsi_matrix, unique)

    data_selected_index = 0
    while(data_selected_index != -1):
        print("\nSelect a tweet ( 0 -", len(data)-1, "): ")
        data_selected_index = int(input())

        if data_selected_index != -1:    
            topics = find_topics(SOM_matrix, lsi_matrix, doc_grams[data_selected_index], unique, lattice_size)

            print("\nRaw Tweet:\n", db_results[data_selected_index]['data']['full_text'])
            print("\nSelected Doc:", doc_grams[data_selected_index])
            print("\nTopics:")
            for topic in topics:
                (location, word) = topic
                (x, y) = location
                print("[", x, "][", y,"] =", word)

    print("---PROGRAM EXITED---")
