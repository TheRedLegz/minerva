from flask import Flask, jsonify, abort, request, abort
from flask_cors import CORS
from modules.tweet_preprocessor import remove_av, remove_hashtags, remove_html_tags, remove_links, remove_non_ascii, remove_users, lower, remove_double_spacing, remove_numbers, remove_punctuations, fix_contractions
from modules.vectorizer import tf_idf, bow
from modules.services import DatabaseConnection
from modules.som import SOM, get_topic_words
from pprint import pprint

app = Flask(__name__)
CORS(app)

db = DatabaseConnection('mongodb://localhost:27017')


# tab 1: unprocessed
# tab 2: preprocessed
# tab 3: grammed

def prepare_tweet(tweet, hasTweetId = True):
    tweet['_id'] = str(tweet['_id'])
    
    if hasTweetId:
        tweet['tweet_id'] = str(tweet['tweet_id'])

    return tweet

def prepare_tweets(arr, hasTweetId = True):
    res = [prepare_tweet(a, hasTweetId) for a in arr]
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
        return jsonify(prepare_tweets(res))

    if isClean == 'true':
        res = list(db.get_clean_tweets()[:20])

    if not res:
        res = list(db.get_full_raw_tweets())[:20]
        return jsonify(prepare_tweets(res))


    return jsonify(prepare_tweets(res))
    

@app.route('/tweets/<tweet_id>')
def get_one_tweet(tweet_id):
    query = request.args

    isFull = str(query.get('full')).lower()
    isClean = str(query.get('clean')).lower()
    
    raw = db.get_tweet_text_by_id(int(tweet_id))
    res = None

    if not raw:
        abort(404)

    if isFull == 'true':
        res = raw

    elif isFull == 'false':
        res = {
            'tweet_id': str(raw['tweet_id']),
            'full_text': raw['data']['full_text']
        }

        return jsonify(res)

    if isClean == 'true':
        res = db.get_one_clean_tweet(int(tweet_id))

    if not res:
        res = raw
        
    return jsonify(prepare_tweet(res))


@app.route('/tweets/<tweet_id>/steps')
def get_one_tweet_pp(tweet_id):
    tweet = db.get_tweet_text_by_id(int(tweet_id))
    
    if not tweet:
        abort(404)

    functions = [remove_users, remove_links, remove_hashtags, remove_av, remove_html_tags, remove_non_ascii, lower, fix_contractions, remove_punctuations, remove_numbers, remove_double_spacing]
    
    steps = [tweet['data']['full_text']]

    for index, method in enumerate(functions):
        steps.append(method(steps[index]))

    res = {
        'tweet_id': str(tweet['data']['id']),
        'full_text': tweet['data']['full_text'],
        'cleaned_tweet': steps[-1:],
        'steps': steps
    }

    return jsonify(res)

@app.route('/vectors')
def get_vectors():
    data = list(db.get_clean_tweets())[:10]
    DOCCOUNT = len(data)

    doc_grams = [a['grams'] for a in data]

    (bowm, unique, _) = bow(doc_grams)
    
    UNIQUE = [a[0] for a in unique]

    res = []

    for idx, doc in enumerate(doc_grams):
        obj = {}
        grams = {}

        for gram in doc:
            try:
                col = UNIQUE.index(gram)
                grams[gram] = int(bowm[idx][col])
            except:
                continue

        obj['grams'] = grams
        obj['full_text'] = data[idx]['full_text']
        obj['tweet_id'] = str(data[idx]['tweet_id'])
        obj['doc_count'] = DOCCOUNT

        res.append(obj)

    return jsonify(res)


@app.route('/vectors/features')
def get_features():

    data = list(db.get_features())
    data = [a['name'] for a in data]
    return jsonify(data)


def tfidf():
    data = list(db.get_clean_tweets())[:10]
    doc_grams = [a['grams'] for a in data]

    (bowm, unique, _) = bow(doc_grams)
    
    unique = [a[0] for a in unique]

    matrix = tf_idf(doc_grams, bowm)

    return (matrix, unique)

@app.route('/som')
def get_som():

    (matrix, unique) = tfidf()
    size = (2,2)

    som = SOM(matrix, 0.1, size)

    return jsonify(get_topic_words(som, unique, size))



    
if __name__ == '__main__':
    app.run(debug=True)
