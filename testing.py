from gettext import find
from minisom import MiniSom
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from modules.services import DatabaseConnection
from modules.gram import tweet_grammer

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

from modules.tweet_preprocessor import clean_documents


db = DatabaseConnection('mongodb://localhost:27017')
raw_tweet = db.get_raw_tweets()
# raw_tweet = [tweet['full_text'] for tweet in raw_tweet]
original_data = [tweet['full_text'] for tweet in raw_tweet]

original = pd.read_csv('tweets_processed.csv')
# original_data = original['Content'][:1000]
# training_data = [document for document in original['Processed']]
# training_data = [ast.literal_eval(document) for document in training_data]
# training_data = [" ".join([doc.strip() for doc in document]) for document in training_data]
# dataset = training_data

# dataset = db.get_preprocessed_tweets()
# dataset = [tweet['preprocessed_text'] for tweet in dataset]

dataset = pd.read_csv('tweets.csv')
dataset = dataset['Tweet Content']
original = dataset
dataset = clean_documents(dataset)


# dataset = dataset + training_data

documents = tweet_grammer(dataset)
documents = [" ".join(document) for document in documents]

# dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                              remove=('headers', 'footers', 'quotes'))
# documents = dataset.data
# doc_targets = dataset.target
# print(dataset.target)

no_features = 1000

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=no_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
D = tfidf.todense().tolist()

n_neurons = 3
m_neurons = 3
som = MiniSom(n_neurons, m_neurons, no_features)
som.train(D, 100)


cluster_matrix = np.empty(shape=(n_neurons, m_neurons), dtype=object)

for i in range(n_neurons):
    for j in range(m_neurons):
        cluster_matrix[i][j] = []

for i in range(len(original)):
    (row, col) = som.winner(D[i])
    cluster_matrix[row][col].append(original[i])


cluster_string = ""

for i in range(n_neurons):
    for j in range(m_neurons):
        cluster_string += "[" + str(i) + "]" + "[" + str(j) + "]" + " = ---NEWLINE--- "
        for document in cluster_matrix[i][j]:
            cluster_string += document
            cluster_string += " ---NEWLINE--- "

# for i in range(n_neurons):
#     for j in range(m_neurons):
#         targets = np.zeros(20, dtype=int) 
#         cluster_string += "[" + str(i) + "]" + "[" + str(j) + "]" + " =\n"
#         for document in cluster_matrix[i][j]:
#             targets[document] += 1
#         for k in range(20):
#             if (targets[k] > 0):
#                 cluster_string += str(k)+ ": " + str(targets[k]) + "\n"
#         cluster_string += "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"

f = open("clusters.txt", "w")
f.write(str(cluster_string.encode("utf8")))
f.close()


# som.pca_weights_init(D)
# som.train(D, 40000, random_order=False, verbose=False)

top_keywords = 10

weights = som.get_weights()
cnt = 1
for i in range(n_neurons):
    for j in range(m_neurons):
        keywords_idx = np.argsort(weights[i,j,:])[-top_keywords:]
        keywords = ' '.join([tfidf_feature_names[k] for k in keywords_idx])
        print('Topic', cnt, ':', keywords)
        cnt += 1
