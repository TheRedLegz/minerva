from this import d
from modules.tweet_preprocessor import basic_clean, prepare_for_chunking
from modules.sentiment import chunker, get_sentiment
from modules.services import DatabaseConnection
from modules.som import tweet_find_cluster
from modules.gram import gram_sentence
from langdetect import detect
import logging as lg
import pickle

db = DatabaseConnection('mongodb://localhost:27017')

TABLE_RAW = 'rawtweets'
TABLE_CLEANED = 'cleaned_tweets'


def load_som():
    somd = db.get_training_model()

    if not somd:
        lg.exception('App must have a model before initialization')
        raise Exception('no_som')

    som = None
    som_size = (somd['size']['row'], somd['size']['col'])

    with open(somd['path'], 'rb') as file:
        som = pickle.load(file)

    if som is None:
        lg.exception('Cannot load SOM')
        raise Exception('som_not_loaded')

    return {
        "model": som,
        "size": som_size,
        "iterations": somd['iterations'],
        "rate": somd['rate'],
    }

def load_features():
    ft = list(db.get_training_features())

    unq_all = []
    idfs = []
    tup = []

    for a in ft:
        unq_all.append(a['name'])
        idfs.append(a['idf'])
        tup.append((a['name'], a['idf']))

    return {
        "features": unq_all,
        "idf": idfs,
        "tup": tup
    }

def init_save_raw(data, scrape_date, scrape_params):
    d = data
    
    d['full_text'] = d['tweet']

    insert = {
        "scrape_date": scrape_date,
        "tweet_id": d['id'],
        "data": d,
        "parameters": scrape_params
    }

    db.upsert_to_collection(TABLE_RAW, { "tweet_id": d['id'] }, insert)
    return insert

def init_save_clean(data):
    
    def execute(raw):
        tt = raw['data']['tweet']
        pp = basic_clean(tt)
        grams = gram_sentence(pp)

        if len(grams) == 0:
            return False

        unq = []

        for g in grams:
            if g not in unq:
                unq.append(g) 

        chunks = chunker(prepare_for_chunking(tt))
        
        chunk_details = []

        for c in chunks:
            res = {}
            gm = gram_sentence(c)

            if len(gm) == 0:
                continue
        
            (_, bmu) = tweet_find_cluster(som, som_size, gm, tup)

            (sentiment, score) = get_sentiment(c)

            res['score'] = score
            res['sentiment'] = sentiment
            res['chunk'] = c
            res['grams'] = gm
            res['cluster'] = {
                "row": bmu[0],
                "col": bmu[1],
                "distance": _[bmu[0],bmu[1]]
            }

            chunk_details.append(res)


        (sent, scr) = get_sentiment(tt)
        

        insert = {
            "tweet_id": raw['tweet_id'],
            "full_text": tt,
            "cleaned": pp,
            "grams": grams,
            "unique_grams": unq,
            "chunk_details": chunk_details,
                "overall_sentiment": {
                'sentiment': sent,
                'score': scr,

            },
            "parameters": raw['parameters']
        }

        db.upsert_to_collection(TABLE_CLEANED, { 'tweet_id': raw['tweet_id'] }, insert)

        return insert

    somd = load_som()
    ft = load_features()
    tup = ft['tup']

    som_size = somd['size']
    som = somd['model']

    if type(data) is list:
        count = 0
        for a in data:
            print(count)
            count += 1
            execute(a)

    return execute(a)


def filter_non_eng(arr):
    res = []
    for r in arr:
        try:
            lang = r['data']['lang'] if 'lang' in r['data'] else detect(r['data']['tweet'])
        except Exception as e:
            print(str(e))
            continue

        if lang.lower() == 'en':
            res.append(r)
        

    return res
