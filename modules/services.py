import pickle
import codecs
from flask import jsonify
from pymongo import MongoClient
import datetime

from regex import P
from modules.gram import tweet_cleaner, tweet_grammer, tweet_pos
from modules.som import SOM
from modules.vectorizer import bow, tf_idf

class DatabaseConnection:
    def __init__(self, db_path):
        self.db_path = db_path  
        self.client = MongoClient(self.db_path)
        self.db_name = 'minerva_raw_tweets'

    def get_raw_tweets(self):
        '''Returns all the raw tweets from the database'''
        data = []
        raw_tweet_collection = list(self.client[self.db_name]['rawtweets'].find())

        for tweet in raw_tweet_collection:
            data.append({'tweet_id': tweet['tweet_id'], 'full_text': tweet['data']['full_text']})

        return data

    def get_preprocessed_tweets(self):
        '''Returns all the preprocessed tweets from the database'''

        data = []
        raw_tweet_collection = list(self.client[self.db_name]['preprocessed_tweets'].find())

        for tweet in raw_tweet_collection:
            data.append({'tweet_id': tweet['tweet_id'], 'preprocessed_text': tweet['preprocessed_text']})

        return data

    def get_unprocessed_tweets(self):
        '''Returns the raw tweets that hasn't been preprocessed yet from the database'''
        data = []
        db = self.client[self.db_name]
        preprocessed_tweet_collection = list(db['preprocessed_tweets'].find())
        preprocessed_tweet_ids = [tweet['tweet_id'] for tweet in preprocessed_tweet_collection]
        unprocessed_tweets = list(db['rawtweets'].find({'tweet_id' : { '$nin' : preprocessed_tweet_ids}}))

        for tweet in unprocessed_tweets:
            data.append({'tweet_id': tweet['tweet_id'], 'full_text': tweet['data']['full_text']})
            
        return data

    def add_to_collection(self, data, collection_name):
        '''Adds data to a collection in the database'''
        if len(data) == 1:
            self.client[self.db_name][collection_name].insert_one(data)
        elif len(data) > 1:
            self.client[self.db_name][collection_name].insert_many(data)
        else:
            print("EXCEPTION: No data to be added")

    def get_tweet_text_by_id_array(self, tweet_id_array):
        tweet_list = list(self.client[self.db_name]['rawtweets'].find({'tweet_id' : { '$in' : tweet_id_array}}))
        tweet_list = [tweet['data']['full_text'] for tweet in tweet_list]
        return tweet_list

    def get_tweet_text_by_id(self, tweet_id):
        tweet = list(self.client[self.db_name]['rawtweets'].find({'tweet_id' : tweet_id}))

        return tweet['data']['full_text']

    def add_model(self):
        raw_tweet_collection = list(self.client[self.db_name]['rawtweets'].find())
        data = []

        for tweet in raw_tweet_collection:
            data.append({'tweet_id': tweet['tweet_id'], 'full_text': tweet['data']['full_text']})

        raw_tweet_collection = data
        data = []
            
        for i in range(10000):
            data.append(raw_tweet_collection[i]['full_text'])

        cleaned = tweet_cleaner(data)

        grammed = tweet_grammer(cleaned)
        for i, tweet_grams in enumerate(grammed):
            grammed[i] = ' '.join(tweet_grams)
        pos_tags = tweet_pos(grammed)
        for i, tweet_grams in enumerate(pos_tags):
            temp = []
            for gram in tweet_grams:
                if gram[1].startswith('NN'):
                    temp.append(gram[0])
            grammed[i] = temp

        (bag, unique, docs) = bow(grammed)

        matrix = tf_idf(docs, bag)

        lattice_size = (4, 4)
        (row, col) = lattice_size

        SOM_matrix = SOM(matrix, .3, lattice_size)

        pickled = codecs.encode(pickle.dumps(SOM_matrix), "base64").decode()
        print(pickled)

        self.client[self.db_name]["models"].insert_one({
            "date" : datetime.datetime.now().timestamp(),
            "model" : pickled,
        })

        return jsonify({
            "response": "Model added successfully"
        });

    def get_model(self):
        print("add model")
