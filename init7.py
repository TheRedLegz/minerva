from modules.lib import load_som, load_features
from modules.services import DatabaseConnection
from modules.som import tweet_find_cluster

db = DatabaseConnection('mongodb://localhost:27017')

som = load_som()
features = load_features()

tup = features['tup']
som_size = som['size']
som = som['model']

res = list(db.get_clean_tweets())

for data in res:
    grams = data['grams']
    (matrix, bmu) = tweet_find_cluster(som, som_size, grams, tup)
    db.conn['cleaned_tweets'].update_one({'tweet_id': data['tweet_id']}, {"$set": {
                                         'cluster': {'row': bmu[0], 'col': bmu[1]}}})
