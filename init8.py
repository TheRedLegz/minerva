from minerva.modules.lib import init_save_clean
from modules.som import SOM, tweet_find_cluster
from modules.services import DatabaseConnection
from modules.vectorizer import bow, tf_idf
import os
import pickle
from datetime import datetime

db = DatabaseConnection('mongodb://localhost:27017')
table = db.conn['training_model']
table2 = db.conn['cleaned_tweets']


tweets = list(db.get_clean_tweets())
grams = []


for idx, doc in enumerate(tweets):
    g = doc['grams']
    temp = [a for a in g if 'learn' not in a and 'education' not in a]
    tweets[idx]['grams'] = temp
    grams.append(grams)


(bowm, unq, _, idf) = bow(grams, 4000, True)

matrix = tf_idf(grams, bowm)

LEARN_RATE = 0.05
SIZE = (4,4)
ITERATIONS = 8000

som = SOM(matrix, LEARN_RATE, SIZE, ITERATIONS)


curpath = os.path.abspath(os.curdir)

file_name = os.path.join(curpath, 'data\models\\training_final(4,4).pkl')

with open(file_name, 'wb') as f:
    pickle.dump(som, f)

insert = {
    "path": file_name,
    "iterations": ITERATIONS,
    "rate": LEARN_RATE,
    "size": {
        "row": SIZE[0],
        "col": SIZE[1],
    },
    "createdAt": datetime.today()
}

table.insert_one(insert)




