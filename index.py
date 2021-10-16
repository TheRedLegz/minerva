from modules.gram import gram_documents
from pprint import pprint as print
import pandas as pd

import numpy as np
import json
from requests.api import get
from modules.tweet_preprocessor import preprocess_documents
from modules.word2vec import get_word2vec_from_data
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
from modules.pca import pca
from modules.lsi import lsi
from matplotlib import pyplot as plt
from modules.som import SOM, find_topics, print_data_to_SOM
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pandas as pd

# client = MongoClient('mongodb://localhost:27017')
# db_raw = client['minerva_raw_tweets']
# rawtweets = db_raw['rawtweets']


if __name__ == "__main__":

    # data = []
    data = pd.read_csv('covid19_tweets.csv')
    data = data['text'].values
    # All tweets
    # for tweet in db_results:
    #     data.append(tweet['tweet'])

    # From MongoDB
    # db_results = list(rawtweets.find())
    # data = []
    # for a in db_results:s
    #     data.append(a['data']['full_text'])

    print("Starting Preprocessing")
    data = preprocess_documents(data[:200])
    print("Finished Preprocessing")
    dataSentiment = preprocess_documents(data[:200])
    #WORD2VEC implementation
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

    # TF-IDF implementation
    bowres = bag_of_words(data, to_preprocess=False)
    (bow, unique, doc_grams) = bowres

    (bow, unique, doc_grams) = prune_bow(bowres, 3)

    vectors = tf_idf(data, bow)

    vectors_t = np.transpose(vectors)
    (lsi_matrix, sum) = pca(vectors_t)
    
    print(lsi_matrix.shape )

    lattice_size = (6, 6)
    (row, col) = lattice_size

    #WORD2VEC implementation
    # SOM_matrix = SOM(vectorized_words, .5, lattice_size)
    #TF-IDF implementation
    SOM_matrix = SOM(lsi_matrix, .5, lattice_size)

    print("\nFinal SOM weights")
    print("Lattice size: (%d, %d)" %(row, col))


    print("\nThe Clustered Topics")
    print_data_to_SOM(SOM_matrix, lsi_matrix, unique)

        if sentiment_score['compound'] >= 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Positive')
        elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Neutral')
        elif sentiment_score['compound'] <= -0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Negative')
    
    sentiRes['sentiment'] = sentiment_label_list
    sentiRes['sentiment score'] = sentiment_score_list

    print(sentiRes)
    
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



    print("---PROGRAM EXITED---")
