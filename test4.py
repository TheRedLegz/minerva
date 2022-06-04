from modules.lib import load_som, load_features
from modules.som import tweet_find_cluster
from modules.services import DatabaseConnection

db = DatabaseConnection('mongodb://localhost:27017')

cleaned_tweets = list(db.get_clean_tweets())[:100]
cleaned_tweets = [tweet['cleaned'] for tweet in cleaned_tweets]
print(cleaned_tweets[1])
SOM_data = load_som()
feature_data = load_features()
feature_data = feature_data['tup']

bmu_list = []

for tweet in cleaned_tweets:
    (_, bmu) = tweet_find_cluster(SOM_data['model'], (SOM_data['row'], SOM_data['col']), tweet, feature_data)
    (row, col) = bmu
    cluster = row * SOM_data['col'] + col
    bmu_list.append(cluster)

print(bmu_list)
