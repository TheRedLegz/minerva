from pymongo import MongoClient
from pprint import pprint

import numpy as np

from modules.preprocessor import preprocess_documents, write_to_file
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
from modules.pca import pca
from matplotlib import pyplot as plt
from modules.som import SOM, find_topics, print_data_to_SOM
client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']


if __name__ == "__main__":

    # Fetching scraped tweets from MongoDB
    db_results = list(rawtweets.find())
    data = []

    print("Started Loading Data")
    # Assigning tweets in variable

    for a in db_results:
        data.append(a['data']['full_text'])

    write_to_file("unprocessed_tweets.txt", data)
    print("Finished Loading Data\n")

    print("Started Preprocessing")
    data = preprocess_documents(data)

    write_to_file("preprocessed_tweets.txt", data)
    print("Finished Preprocessing\n")

    print("Started BOW")
    bowres = bag_of_words(data, to_preprocess=False)
    (bow, unique, doc_grams) = bowres

    write_to_file("raw_bow.txt", bow, is_2d=True)
    write_to_file("raw_unique_words.txt", unique)
    write_to_file("raw_doc_grams.txt", doc_grams, is_2d=True)
    print("Finished BOW\n")

    print("Started Pruning")
    (bow, unique, doc_grams) = prune_bow(bowres)

    write_to_file("pruned_bow.txt", bow, is_2d=True)
    write_to_file("pruned_unique_words.txt", unique)
    write_to_file("pruned_doc_grams.txt", doc_grams, is_2d=True)
    print("Finished Pruning\n")

    print("Started TF-IDF")
    vectors = tf_idf(data, bow)

    np.savetxt("tf_idf_raw.csv", vectors, delimiter=',')
    write_to_file("tf_idf.txt", vectors, is_2d=True)

    vectors_t = np.transpose(vectors)

    np.savetxt("tf_idf_transposed.csv", vectors_t, delimiter=',')
    write_to_file("tf_idf_transposed.txt", vectors_t, is_2d=True)
    print("Finished TF-IDF\n")

    print("Started PCA")
    (pca_matrix, sum) = pca(vectors_t)
    
    print(pca_matrix.shape )
    np.savetxt("pca_data.csv", pca_matrix, delimiter=',')
    write_to_file("pca.txt", pca_matrix, is_2d=True)
    print("Finished PCA\n")

    optimal_i = np.arange(len(sum)) + 1
    plt.bar(optimal_i ,sum)
    plt.title('Best Number of Features')
    plt.xlabel('Features')
    plt.ylabel('Explained Variance Ratio')
    plt.show()

    lattice_size = (3, 3)
    (row, col) = lattice_size

    print("Started SOM")
    SOM_matrix = SOM(pca_matrix,.5, lattice_size)
    print("Finished SOM")
    
    print("\nFinal SOM weights")
    print("Lattice size: (%d, %d)" %(row, col))

    for i in range(row):
        for j in range(col):
            print("[", i, "] [", j, "] =", SOM_matrix[i][j][:3])

    print("\nThe Clustered Topics")
    print_data_to_SOM(SOM_matrix, pca_matrix, unique)

    data_selected_index = 0
    while(data_selected_index != -1):
        print("\nSelect a tweet ( 0 -", len(data)-1, "): ")
        data_selected_index = int(input())

        if data_selected_index != -1:    
            topics = find_topics(SOM_matrix, pca_matrix, doc_grams[data_selected_index], unique, lattice_size)

            print("\nRaw Tweet:\n", db_results[data_selected_index]['data']['full_text'])
            print("\nSelected Doc:", doc_grams[data_selected_index])
            print("\nTopics:")
            for topic in topics:
                (location, word) = topic
                (x, y) = location
                print("[", x, "][", y,"] =", word)

    print("---PROGRAM EXITED---")
