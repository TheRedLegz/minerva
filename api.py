from flask import Flask, jsonify, abort
from pymongo import MongoClient
from flask_cors import CORS
from modules.tweet_preprocessor import basic_clean, preprocess_tweet
from modules.gram import gram_sentence
from modules.sentiment import sentimentinator

import numpy as np
from modules.tweet_preprocessor import preprocess_documents
from modules.word2vec import get_word2vec_from_data
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
from modules.pca import pca
from modules.lsi import lsi
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
db_results_model = list(rawtweets.find())
data_model = []
for a in db_results_model:
    data_model.append(a['data']['full_text'])

print("Starting Preprocessing")
data = preprocess_documents(data_model[:200])
print("Finished Preprocessing")

# dataSentiment = preprocess_documents(data[:200])

#WORD2VEC implementation
model = get_word2vec_from_data(data, to_preprocess=False)

vectorized_words = []
unique = []
for i in range(len(model.wv.index_to_key)):
    word = model.wv.index_to_key[i]
    unique.append(word)
    vectorized_words.append(model.wv[word])
vectorized_words = np.asarray(vectorized_words)

doc_grams = []
for sentence in data:
    doc_grams.append([word for word in sentence if word in unique])

# TF-IDF implementation
# bowres = bag_of_words(data, to_preprocess=False)
# (bow, unique, doc_grams) = bowres

# (bow, unique, doc_grams) = prune_bow(bowres, 3)

# vectors = tf_idf(data, bow)

# vectors_t = np.transpose(vectors)
# (lsi_matrix, sum) = pca(vectors_t)

# print(lsi_matrix.shape )

lattice_size = (6, 6)
(row, col) = lattice_size

#WORD2VEC implementation
SOM_matrix = SOM(vectorized_words, .5, lattice_size)
#TF-IDF implementation
# SOM_matrix = SOM(lsi_matrix, .5, lattice_size)

print("\nFinal SOM weights")
print("Lattice size: (%d, %d)" %(row, col))


print("\nThe Clustered Topics")
model = get_SOM_model(SOM_matrix, vectorized_words, unique)


# END OF TEMP MODEL CREATOR


# experimental
@app.route('/data', methods=['GET'])
def get_data():
    db_results = list(rawtweets.find())
    data = []

    for a in db_results:
        to_add = a['data']

        to_add['preprocessed'] = preprocess_tweet(a['data']['full_text'])
        grams = gram_sentence(a['data']['full_text'])
        
        to_add['unigrams'] = [token for token in grams if '_' not in token]
        to_add['bigrams'] = [token for token in grams if token.count('_') == 1]
        to_add['trigrams'] = [token for token in grams if token.count('_') == 2]

        data.append(to_add)


    s_data = sentimentinator([item['preprocessed'] for item in data])


    for i, _ in enumerate(data):
        data[i]['sentiment_score'] = s_data.iloc[i]['sentiment_score']
        data[i]['sentiment'] = s_data.iloc[i]['sentiment']


    return jsonify(data)


@app.route('/tweets', methods=['GET'])
def get_tweets():
    db_results = list(rawtweets.find())
    data = []

    for a in db_results:
        data.append(a['data'])

    return jsonify(data)


@app.route('/model', methods=['GET'])
def get_model():

    modelObject = {}

    for i in range(row):
        db_results = list(rawtweets.find())

        modelObject[str(i)] = {}
        for j in range(col):
            modelObject[str(i)][str(j)] = {}
            modelObject[str(i)][str(j)]['keywords'] = model[i][j]
            modelObject[str(i)][str(j)]['tweets'] = []
    for tweet_data in db_results:
        tweet = {}
        preprocessed_tweet_text = tweet_data['data']['full_text']
        preprocessed_tweet_tokens = preprocess_tweet(preprocessed_tweet_text).split(' ')
        tweet['id']= tweet_data['data']['id']
        tweet['tokens'] = preprocessed_tweet_tokens

        for i in range(row):
            for j in range(col):
                intersection = [keyword for keyword in model[i][j] if keyword in preprocessed_tweet_tokens]
                if len(intersection) > 0:
                    modelObject[str(i)][str(j)]['tweets'].append(tweet)
            
    
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
