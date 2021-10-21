from flask import Flask, jsonify, abort
from pymongo import MongoClient
from flask_cors import CORS
from modules.tweet_preprocessor import basic_clean
from modules.gram import gram_documents

app = Flask(__name__)
CORS(app)


client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']


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
    

@app.route('/preprocessing', methods=['GET'])
def get_processing_data():
    db_results = list(rawtweets.find())
    data = []

    for a in db_results:
        to_add = {}
        to_add['full_text'] =  a['data']['full_text']
        to_add['created_at'] =  a['data']['created_at']
        to_add['id'] =  a['data']['id']

        to_add['cleaned'] = basic_clean(a['data']['full_text'])

        data.append(to_add)


    return jsonify({ 'data': data })


@app.route('/grams', methods=['GET'])
def get_grams():
    db_results = list(rawtweets.find())
    data = []

    for a in db_results:
        data.append(a['data']['full_text'])

    tokens = gram_documents(data)

    return jsonify({ 'data': tokens })


if __name__ == '__main__':
    app.run(debug=True)
