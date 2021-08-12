from pymongo import MongoClient
from pprint import pprint

from modules.preprocessor import preprocess_documents, write_to_file
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
from modules.pca import pca
# from modules.som import SOM
from matplotlib import pyplot as plt

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

    (bow, unique, doc_grams) = prune_bow(bowres)

    vectors = tf_idf(data, bow)

    (pca_matrix, sum) = pca(vectors)
