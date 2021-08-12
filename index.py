from pymongo import MongoClient
from pprint import pprint

import numpy as np

from modules.preprocessor import preprocess_documents, write_to_file
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
from modules.pca import pca
from matplotlib import pyplot as plt
from modules.som import SOM, print_data_to_SOM

client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']


if __name__ == "__main__":
    db_results = list(rawtweets.find())
    data = []


    for a in db_results:
        data.append(a['data']['full_text'])
    # write_to_file("unprocessed_tweets.txt", data)

    data = preprocess_documents(data)
    # write_to_file("preprocessed_tweets.txt", data)

    bowres = bag_of_words(data, to_preprocess=False)
    (bow, unique, doc_grams) = bowres
    # write_to_file("raw_bow.txt", bow, is_2d=True)
    # write_to_file("raw_unique_words.txt", unique)
    write_to_file("raw_doc_grams.txt", doc_grams, is_2d=True)

    (bow, unique, doc_grams) = prune_bow(bowres)
    # write_to_file("pruned_bow.txt", bow, is_2d=True)
    # write_to_file("pruned_unique_words.txt", unique)
    write_to_file("pruned_doc_grams.txt", doc_grams, is_2d=True)

    vectors = tf_idf(data, bow)
    # np.savetxt("tf_idf_raw.csv", vectors, delimiter=',')
    # write_to_file("tf_idf.txt", vectors, is_2d=True)
    vectors_t = np.transpose(vectors)
    # np.savetxt("tf_idf_transposed.csv", vectors_t, delimiter=',')
    # write_to_file("tf_idf_transposed.txt", vectors_t, is_2d=True)

    (pca_matrix, sum) = pca(vectors_t)
    print(pca_matrix.shape )
    # np.savetxt("pca_data.csv", pca_matrix, delimiter=',')
    # write_to_file("pca.txt", pca_matrix, is_2d=True)

    # optimal_i = np.arange(len(sum))
    # plt.bar(optimal_i ,sum)
    # plt.title('Best Number of Features')
    # plt.xlabel('Features')
    # plt.ylabel('Explained Variance Ratio')
    # plt.show()

    SOM_matrix = SOM(pca_matrix,.5,(3, 4))
    print_data_to_SOM(SOM_matrix, pca_matrix, unique)
