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

    data = preprocess_documents(data)

    bowres = bag_of_words(data, to_preprocess=False)
    (bow, unique, doc_grams) = bowres
    print(len(unique))

    (bow, unique, doc_grams) = prune_bow(bowres)
    print(len(unique))
    pprint(unique)

    vectors = tf_idf(data, bow)
    vectors_t = np.transpose(vectors)
    print(vectors.shape)
    print(vectors_t.shape)

    (pca_matrix, sum) = pca(vectors_t)

    SOM(pca_matrix,.5,(3, 3))
