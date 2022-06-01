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
        self.conn = self.client[self.db_name]

    def get_full_raw_tweets(self):
        '''Returns all the raw tweets from the database with metadata'''
        return list(self.client[self.db_name]['rawtweets'].find())

    def get_raw_tweets(self):
        '''Returns all the raw tweets from the database'''
        data = []
        raw_tweet_collection = list(self.client[self.db_name]['rawtweets'].find())

        for tweet in raw_tweet_collection:
            data.append({'tweet_id': tweet['tweet_id'], 'full_text': tweet['data']['full_text']})

        return data

    def get_clean_tweets(self):
        '''Returns all the preprocessed tweets from the database'''
        return self.client[self.db_name]['cleaned_tweets'].find()
    
    def get_one_clean_tweet(self, id):
        return self.conn['cleaned_tweets'].find_one({ 'tweet_id': id })

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

    def upsert_to_collection(self, table, query, data):
        self.conn[table].replace_one(query, data, upsert=True)
        return True

    def get_tweet_text_by_id_array(self, tweet_id_array):
        tweet_list = list(self.client[self.db_name]['rawtweets'].find({'tweet_id' : { '$in' : tweet_id_array}}))
        tweet_list = [tweet['data']['full_text'] for tweet in tweet_list]
        return tweet_list

    def get_tweet_text_by_id(self, tweet_id):
        return self.client[self.db_name]['rawtweets'].find_one({'tweet_id' : tweet_id})

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

        # join the grams to form a single sentence
        for i, tweet_grams in enumerate(grammed):
            grammed[i] = ' '.join(tweet_grams)

        pos_tags = tweet_pos(grammed)
        
        # filter out the non-nouns in the grams
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
        })

    def get_model(self):
        return self.conn['models'].find_one()

    def get_features(self):
        return self.conn['features'].find()

    def get_vectors(self):
        return self.conn['vectors'].find()
    
    def get_settings(self):
        return self.conn['settings'].find_one()




    def save_snapshot(self, matrix, step, id):

       model = codecs.encode(pickle.dumps(matrix), "base64").decode()

       self.conn['snapshots'].insert_one({
           "data": model,
           "iteration": step,
           "modelId": id
       })

    def get_snapshots(self):
        return self.conn['snapshots'].find()


    def save_settings(self, size, iterations, rate):
        self.conn['settings'].insert_one({
            'size': size,
            'iterations': iterations,
            'rate': rate,
        })

    def get_scrape_results(self):
        return self.conn['scrape_results'].find()

    def get_training_model(self):
        return self.conn['training_model'].find_one()

    def get_one_vector(self, id):
        return self.conn['vectors'].find_one({ "tweet_id" : id })

    def get_training_features(self):
        return self.conn['training_features'].find()

    def get_training_cleaned(self):
        return self.conn['training_clean'].find()