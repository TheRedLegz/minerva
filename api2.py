from datetime import datetime
from flask import Flask, jsonify, abort, request, abort
from flask_cors import CORS
from modules.som import tweet_find_cluster
from modules.tweet_preprocessor import remove_av, remove_hashtags, remove_html_entities, remove_links, remove_non_ascii, remove_noneng_stopwords, remove_users, lower, remove_double_spacing, remove_numbers, remove_punctuations, fix_contractions
from modules.vectorizer import tf_idf, bow
from modules.services import DatabaseConnection
from modules.som import SOM, get_topic_words
from modules.sentiment import get_sentiment

from pprint import pprint
import asyncio
from modules.scraper import Scraper
import pickle
import math

app = Flask(__name__)
CORS(app)

db = DatabaseConnection('mongodb://localhost:27017')
sc = Scraper()

def prepare_tweet(tweet, hasTweetId=True, hasId=True):
    if hasId:
        tweet['_id'] = str(tweet['_id'])

    if hasTweetId:
        tweet['tweet_id'] = str(tweet['tweet_id'])

    return tweet


def prepare_tweets(arr, hasTweetId=True, hasId=True):
    res = [prepare_tweet(a, hasTweetId, hasId) for a in arr]
    return res


def inject_sentiments(arr):
    for a in arr:
        d = a['full_text'] if 'full_text' in a else a['data']['full_text']
        (sentiment, score) = get_sentiment(d)
        
        a['overall_sentiment'] = {
            "score": score,
            "sentiment": sentiment
        }

    return arr


@app.route('/tweets')
def get_all():
    query = request.args

    isFull = str(query.get('full')).lower()
    isClean = str(query.get('clean')).lower()
    res = None

    if isFull == 'true':
        res = list(db.get_full_raw_tweets())
        res = inject_sentiments(res)
        
    elif isFull == 'false':
        res = list(db.get_raw_tweets())
        res = inject_sentiments(res)
        return jsonify(prepare_tweets(res, True, False))

    if isClean == 'true':
        res = list(db.get_clean_tweets())

    if not res:
        res = list(db.get_full_raw_tweets())
        res = inject_sentiments(res)
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

    functions = [remove_users, remove_links, remove_hashtags, remove_av, remove_html_entities, remove_non_ascii,
                 lower, fix_contractions, remove_punctuations, remove_numbers, remove_double_spacing, remove_noneng_stopwords]

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
    data = list(db.get_vectors())[:10]

    return jsonify(prepare_tweets(data))


@app.route('/vectors/<id>')
def get_one_vector(id):
    data = db.get_one_vector(int(id))

    return jsonify(prepare_tweet(data))


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


# som endpoint should also display the results for every iteration

@app.route('/som/snapshots')
def get_som():
    data = list(db.get_snapshots())
    return jsonify(prepare_tweets(data, False))



def load_som():
    row = db.get_training_model()

    if not row:
        raise Exception('no_som')

    som = None

    with open(row['path'], 'rb') as file:
        som = pickle.load(file)

    if som is None:
        raise Exception('som_not_loaded')

    return {
        "model": som,
        "row": row['size']['row'],
        "col": row['size']['col'],
        "iterations": row['iterations'],
        "rate": row['rate']
    }


def load_training_features():
    features = list(db.get_training_features())

    uniq_tuple = []
    uniq = []

    for a in features:
        uniq.append(a['name'])
        uniq_tuple.append((a['name'], a['idf']))

    return {
        "idf_tuple": uniq_tuple,
        "features": uniq
    }

@app.route('/som/cluster')
def get_clusters():
    try:
        data = load_som()
        som = data['model']

        SIZE = (data['row'], data['col'])
        ft = load_training_features()

        cluster_details = get_topic_words(som, ft['features'], SIZE)

        return jsonify({
            "clusters": cluster_details
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        })

@app.route('/som/cluster/<int:id>')
def get_cluster_details(id):
    try:
        data = load_som()
        som = data['model']
        
        cells = data['row'] * data['col']

        SIZE = (data['row'], data['col'])

        if id >= cells:
            raise Exception('cluster_invalid')

        ft = load_training_features()

        cluster_details = get_topic_words(som, ft['features'], SIZE)

        return jsonify({
           "top_words": cluster_details[id]
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e)
        })
    

@app.route('/som/tweet/<id>')
def get_tweet_cluster(id):
    row = db.get_training_model()
    tweet = db.get_one_clean_tweet(int(id))

    if not row:
        return jsonify({
            "error": "no_som"
        })

    if not tweet:
        return jsonify({
            "error": "no_tweet"
        })

    som = None

    with open(row['path'], 'rb') as file:
        som = pickle.load(file)

    if som is None:
        return jsonify({
            "error": "som_not_loaded"
        })
    
    size = (row['size']['row'], row['size']['col'])

    features = list(db.get_training_features())

    uniq_tuple = []
    uniq = []

    for a in features:
        uniq.append(a['name'])
        uniq_tuple.append((a['name'], a['idf']))
    
    (_, bmu) = tweet_find_cluster(som, size, tweet['grams'], uniq_tuple)

    cluster_details = get_topic_words(som, uniq, size)
    cell = bmu[0] * size[0] + bmu[1]

    top_words = cluster_details[cell]

    return jsonify({
        "row": bmu[0],
        "col": bmu[1],
        "distance": _[bmu[0]][bmu[1]],
        "cluster_details": {
            "top_words": top_words
        }
    })



@app.route('/scrape')
async def trigger_scrape():

    async def scrape():
        await sc.scrape()

    asyncio.create_task(scrape())

    return jsonify({})

@app.route('/scrape/results')
def get_scrape_results():
    results = list(db.get_scrape_results())
    results['_id'] = str(results['_id'])
    
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)

