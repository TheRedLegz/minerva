from pymongo import MongoClient
from pprint import pprint

from modules.preprocessor import preprocess_documents
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
from modules.pca import pca
# from modules.som import SOM


client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']


if __name__ == "__main__":
    db_results = list(rawtweets.find())
    data = []


    for a in db_results:
        data.append(a['data']['full_text'])

    print(len(data))

    data = preprocess_documents(data)
    (bow, unique, doc_grams) = bag_of_words(data, to_preprocess=False)
    pprint(unique)

    (bow, unique, doc_grams) = prune_bow((bow, unique, doc_grams))
    pprint(unique)

    print(bow.shape)
    vectors = tf_idf(data, bow)
    print(vectors.shape)
    
    pca_matrix = pca(vectors)


    pprint(pca_matrix)