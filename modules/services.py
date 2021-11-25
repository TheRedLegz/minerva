from pymongo import MongoClient


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


