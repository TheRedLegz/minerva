from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
# from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from pprint import pprint
import math
# nltk.download('stopwords')

tweets_df = pd.read_json (r'sample.json')

def preprocessing():
    # stop_words = stopwords.words('english')
    tweets_df['clean_doc'] = tweets_df['full_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    tweets_df['clean_doc'] = tweets_df['clean_doc'].str.replace("[^a-zA-Z#]", " ", regex=True)
    tweets_df['clean_doc'] = tweets_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    tweets_df['clean_doc'] = tweets_df['clean_doc'].apply(lambda x: x.lower())
    tokenized_doc = tweets_df['clean_doc'].apply(lambda x: x.split())
    # tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    detokenized_doc = []
    for i in range(len(tweets_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)

    # tweets_df['clean_doc'] = detokenized_doc
    return detokenized_doc


# def bgram(document_array):
#     vectorizer = TfidfVectorizer(ngram_range=(2,2), sublinear_tf=True, use_idf=True)
#     sparse_matrix = vectorizer.fit_transform(document_array)
#     features = vectorizer.get_feature_names()

#     sum = sparse_matrix.sum(axis=0)
#     data = []

#     for i, j in enumerate(features):
#         data.append((j, sum[0, i]))

#     return data

# TODO test accuracy of LSA
# TODO try connecting LSA to SOM
# TODO make SOM

def topicModeling(document_array):
    vectorizer = TfidfVectorizer(ngram_range=(2,2), stop_words="english", max_features= 1000, sublinear_tf=True)

    X = vectorizer.fit_transform(document_array)
    features = vectorizer.get_feature_names()

    svd_model = TruncatedSVD(n_components=26, n_iter=100)
    svd_model.fit(X)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(svd_model.components_)
    # principalDf = pd.DataFrame(data=principalComponents)

    return principalComponents


    # return principalDf





# def topicModeling():
#     vectorizer = TfidfVectorizer(stop_words='english', 
#     max_features= 1000, # keep top 1000 terms 
#     max_df = 0.5, 
#     smooth_idf=True)
#     X = vectorizer.fit_transform(tweets_df['clean_doc'])
#     # print(X.shape)

#     svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
#     svd_model.fit(X)
#     terms = vectorizer.get_feature_names()

#     # print(len(terms))
#     # for i, comp in enumerate(svd_model.components_):
#     #     terms_comp = zip(terms, comp)
#     #     sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
#         # print("Topic "+str(i)+": ")
#         # for t in sorted_terms:
#         #     print(t)
#         #     print(" ")
#         # print(sorted_terms)

#     pca = PCA(n_components = 2)
#     principalComponents = pca.fit_transform(svd_model.components_)
#     principalDf = pd.DataFrame(data = principalComponents, columns = ['1', '2'])
    
#     return principalDf


# texts = preprocessing()
# res = topicModeling(texts)

# pprint(res)
