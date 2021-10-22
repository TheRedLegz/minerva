from flask import Flask, jsonify, abort
from pymongo import MongoClient
from flask_cors import CORS
from modules.tweet_preprocessor import basic_clean, preprocess_tweet
from modules.gram import gram_sentence
from modules.sentiment import sentimentinator

app = Flask(__name__)
CORS(app)


client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']



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
