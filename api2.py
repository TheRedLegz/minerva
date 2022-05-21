from flask import Flask, jsonify, abort, request, abort
from flask_cors import CORS
from modules.tweet_preprocessor import remove_av, remove_hashtags, remove_html_tags, remove_links, remove_non_ascii, remove_users, lower, remove_double_spacing, remove_numbers, remove_punctuations, fix_contractions

from modules.services import DatabaseConnection

app = Flask(__name__)
CORS(app)

db = DatabaseConnection('mongodb://localhost:27017')


# tab 1: unprocessed
# tab 2: preprocessed
# tab 3: grammed

def prepare_tweet(tweet):
    tweet['_id'] = str(tweet['_id'])
    return tweet

def prepare_tweets(arr):
    res = [prepare_tweet(a) for a in arr]
    return res

@app.route('/tweets')
def get_all():
    query = request.args

    isFull = str(query.get('full')).lower()
    isClean = str(query.get('clean')).lower()
    res = None

    if isFull == 'true':
        res = list(db.get_full_raw_tweets())[:20]

    elif isFull == 'false':
        res = list(db.get_raw_tweets())[:20]
        return jsonify(res)

    if isClean == 'true':
        res = list(db.get_clean_tweets()[:20])

    if not res:
        res = list(db.get_raw_tweets())[:20]
        return jsonify(res)


    return jsonify(prepare_tweets(res))
    

@app.route('/tweets/<tweet_id>')
def get_one_tweet(tweet_id):
    query = request.args

    isFull = str(query.get('full')).lower()
    isClean = str(query.get('clean')).lower()
    
    raw = db.get_tweet_text_by_id(tweet_id)

    if not raw:
        abort(404)

    if isFull == 'true':
        return jsonify(raw)

    elif isFull == 'false':
        res = {
            'tweet_id': raw['tweet_id'],
            'full_text': raw['data']['full_text']
        }

        return jsonify(res)

    if isClean == 'true':
        return jsonify(db.get_one_clean_tweet(tweet_id))

    return raw


@app.route('/tweets/<tweet_id>/steps')
def get_one_tweet_pp(tweet_id):
    tweet = db.get_tweet_text_by_id(int(tweet_id))
    
    if not tweet:
        abort(404)

    functions = [remove_users, remove_links, remove_hashtags, remove_av, remove_html_tags, remove_non_ascii, lower, fix_contractions, remove_punctuations, remove_double_spacing, remove_numbers]
    
    steps = [tweet['data']['full_text']]

    for index, method in enumerate(functions):
        steps.append(method(steps[index]))

    res = {
        'tweet_id': tweet['data']['id'],
        'full_text': tweet['data']['full_text'],
        'cleaned_tweet': steps[-1:],
        'steps': steps
    }

    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)
