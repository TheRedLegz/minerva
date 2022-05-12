from flask import Flask, jsonify, abort
from flask_cors import CORS
from pymongo import MongoClient
from modules.tweet_preprocessor import preprocess_tweet
from modules.gram import gram_sentence, gram_documents
from modules.sentiment import sentimentinator
from gensim.corpora import Dictionary
from modules.vectorizer import bow, tf_idf
import numpy as np

import numpy as np
import concurrent.futures
from modules.tweet_preprocessor import preprocess_documents
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
# from modules.pca import pca
# from modules.lsi import lsi
from matplotlib import pyplot as plt
from modules.som import SOM, find_topics, get_SOM_model, print_data_to_SOM
from modules.sentiment import sentimentinator

app = Flask(__name__)
CORS(app)

client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']

# TEMPORARY MODEL CREATOR ON RUN

# From MongoDB
# db_results_model = list(rawtweets.find())
# data_model = []
# for a in db_results_model:
#     data_model.append(a['data']['full_text'])

# print("Starting Preprocessing")
# data = preprocess_documents(data_model[:200])
# print("Finished Preprocessing")

# # dataSentiment = preprocess_documents(data[:200])

# #WORD2VEC implementation
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

# # TF-IDF implementation
# # bowres = bag_of_words(data, to_preprocess=False)
# # (bow, unique, doc_grams) = bowres

# # (bow, unique, doc_grams) = prune_bow(bowres, 3)

# # vectors = tf_idf(data, bow)

# # vectors_t = np.transpose(vectors)
# # (lsi_matrix, sum) = pca(vectors_t)

# # print(lsi_matrix.shape )

# lattice_size = (6, 6)
# (row, col) = lattice_size

# #WORD2VEC implementation
# SOM_matrix = SOM(vectorized_words, .5, lattice_size)
# #TF-IDF implementation
# # SOM_matrix = SOM(lsi_matrix, .5, lattice_size)

# print("\nFinal SOM weights")
# print("Lattice size: (%d, %d)" %(row, col))


# print("\nThe Clustered Topics")
# model = get_SOM_model(SOM_matrix, vectorized_words, unique)


# END OF TEMP MODEL CREATOR

# experimental
@app.route('/data', methods=['GET'])
def get_data():
    db_results = list(rawtweets.find()[:1000])
    data = []

    divisions = int(len(db_results) / 4)
    data_array = [] 
    data_array.append(db_results[0:divisions])
    data_array.append(db_results[divisions:divisions*2])
    data_array.append(db_results[divisions*2:divisions*3])
    data_array.append(db_results[divisions*3:])

    def _append_preprocessed_data(array):
        temp = []
        for a in array:
            to_add = a['data']

            to_add['preprocessed'] = preprocess_tweet(a['data']['full_text'])
            grams = gram_sentence(to_add['preprocessed'])
            
            to_add['unigrams'] = [token for token in grams if '_' not in token]
            to_add['bigrams'] = [token for token in grams if token.count('_') == 1]
            to_add['trigrams'] = [token for token in grams if token.count('_') == 2]
            to_add['tokens'] = to_add['unigrams'] + to_add['bigrams'] + to_add['trigrams']

            temp.append(to_add)


        s_data = sentimentinator([item['preprocessed'] for item in temp])

        for i, _ in enumerate(data):
            temp[i]['sentiment_score'] = s_data.iloc[i]['sentiment_score']
            temp[i]['sentiment'] = s_data.iloc[i]['sentiment']

        return temp

    # for a in db_results:
    #     to_add = a['data']

    #     to_add['preprocessed'] = preprocess_tweet(a['data']['full_text'])
    #     grams = gram_sentence(to_add['preprocessed'])
        
    #     to_add['unigrams'] = [token for token in grams if '_' not in token]
    #     to_add['bigrams'] = [token for token in grams if token.count('_') == 1]
    #     to_add['trigrams'] = [token for token in grams if token.count('_') == 2]
    #     to_add['tokens'] = to_add['unigrams'] + to_add['bigrams'] + to_add['trigrams']

    #     data.append(to_add)

        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tempData = []
        for result in executor.map(_append_preprocessed_data, data_array):
            tempData.append(result)

        for array in tempData:
            data = data + array


    s_data = sentimentinator([item['preprocessed'] for item in data])


    for i, _ in enumerate(data):
        data[i]['sentiment_score'] = s_data.iloc[i]['sentiment_score']
        data[i]['sentiment'] = s_data.iloc[i]['sentiment']


    return jsonify(data)

@app.route('/tokens', methods=['GET'])
def get_tokens():
    db_results = list(rawtweets.find())
    data = [item['data']['full_text'] for item in db_results]

    cleaned = gram_documents(data)
    dt = Dictionary(cleaned)

    uni = {}
    bi = {}
    tri = {}

    cfs = dt.cfs
    dfs = dt.dfs

    res = {}

    frequencies = {}

    for word, id in dt.token2id.items():
        frequencies[word] = {}

        frequencies[word]['cfs'] = cfs[id]
        frequencies[word]['dfs'] = dfs[id]

        if '_' not in word:
            uni[word] = id
        elif word.count('_') == 1:
            bi[word] = id
        elif word.count('_') == 2:
            tri[word] = id

    res['uni'] = uni
    res['bi'] = bi
    res['tri'] = tri
    res['frequencies'] = frequencies

    return jsonify(res)


@app.route('/tfidf', methods=['GET'])
def get_vectors():
    db_results = list(rawtweets.find())
    data = [item['data']['full_text'] for item in db_results]


    cleaned = gram_documents(data)

    (bows, unique) = bow(cleaned)
    matrix = tf_idf(cleaned, bows)

    flat_matrix = np.array(matrix).flatten()

    docs = {}

    for i, doc in enumerate(cleaned):
        docs[i] = doc

    res = {
        'cols': unique,
        'tfidf': list(flat_matrix),
        'docs': docs
    }

    return jsonify(res)
    
@app.route('/tweets', methods=['GET'])
def get_tweets():
    db_results = list(rawtweets.find())
    data = []

    for a in db_results:
        data.append(a['data'])

    return jsonify(data)


# @app.route('/model', methods=['GET'])
# def get_model():

#     modelObject = {}

#     for i in range(row):
#         db_results = list(rawtweets.find())

#         modelObject[str(i)] = {}
#         for j in range(col):
#             modelObject[str(i)][str(j)] = {}
#             modelObject[str(i)][str(j)]['keywords'] = model[i][j]
#             modelObject[str(i)][str(j)]['tweets'] = []
#     for tweet_data in db_results:
#         tweet = {}
#         preprocessed_tweet_text = tweet_data['data']['full_text']
#         preprocessed_tweet_tokens = preprocess_tweet(preprocessed_tweet_text).split(' ')
#         tweet['id']= tweet_data['data']['id']
#         tweet['tokens'] = preprocessed_tweet_tokens

#         for i in range(row):
#             for j in range(col):
#                 intersection = [keyword for keyword in model[i][j] if keyword in preprocessed_tweet_tokens]
#                 if len(intersection) > 0:
#                     modelObject[str(i)][str(j)]['tweets'].append(tweet)
            
    
    return jsonify(modelObject)

@app.route('/tweets/<int:tweet_id>')
def get_one_tweet(tweet_id):

    res = rawtweets.find_one({ 'data': { 'id': tweet_id }})
    
    error = jsonify({
        'error': 'Tweet does not exist'
    })

    if res:
        return jsonify(res)

    return error
    
if __name__ == '__main__':
    app.run(debug=True)
