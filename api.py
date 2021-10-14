from flask import Flask, jsonify, abort
from pymongo import MongoClient

app = Flask(__name__)
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
    
if __name__ == '__main__':
    app.run(debug=True)
