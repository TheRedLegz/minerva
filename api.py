from flask import Flask, jsonify, abort
from pymongo import MongoClient
from flask_cors import CORS
from modules.tweet_preprocessor import preprocess_tweet
from modules.gram import gram_sentence, gram_documents
from modules.sentiment import sentimentinator
from gensim.corpora import Dictionary
from modules.vectorizer import bow, tf_idf
import numpy as np

app = Flask(__name__)
CORS(app)


client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']


# experimental
@app.route('/data', methods=['GET'])
def get_data():
    db_results = list(rawtweets.find()[:1000])
    data = []

    for a in db_results:
        to_add = a['data']

        to_add['preprocessed'] = preprocess_tweet(a['data']['full_text'])
        grams = gram_sentence(to_add['preprocessed'])
        
        to_add['unigrams'] = [token for token in grams if '_' not in token]
        to_add['bigrams'] = [token for token in grams if token.count('_') == 1]
        to_add['trigrams'] = [token for token in grams if token.count('_') == 2]
        to_add['tokens'] = to_add['unigrams'] + to_add['bigrams'] + to_add['trigrams']

        data.append(to_add)


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
