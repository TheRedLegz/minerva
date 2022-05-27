from datetime import datetime
from modules.gram import tweet_grammer
from modules.sentiment import chunker
from modules.services import DatabaseConnection
from modules.sentiment import get_sentiment, chunker
from modules.tweet_preprocessor import prepare_for_chunking
from modules.vectorizer import bow, tf_idf
from modules.som import SOM

import json
import os

import pickle
import codecs
import pandas as pd

db = DatabaseConnection('mongodb://localhost:27017')
table = db.conn['training_clean']
table2 = db.conn['training_model']
table3 = db.conn['training_features']

df = pd.read_csv('./data/tweets_processed.csv')
datarows = list(df.iterrows())[:200]


dataclean = [json.loads(a.replace("'", "\"")) for a in list(df['Processed'])[:200]]

print(dataclean[:3])


# training_clean
for i, row in datarows:
    full_text = row['Content']

    chunks = chunker(prepare_for_chunking(full_text))

    chunk_details = []

    for a in chunks:
        (sent, score) = get_sentiment(a)
        res = {
            "chunk": a,
            "score": score,
            "sentiment": sent
        }
        chunk_details.append(res)
        
    (sent, score) = get_sentiment(full_text)

    doc_uniq = []

    for a in list(row['Processed']):
        if a not in doc_uniq:
            doc_uniq.append(a)
        

    insert_to_cleaned = {
        "grams": list(row['Processed']),
        "full_text": full_text,
        "chunk_details": chunk_details,
        "overall_sentiment": {
            "score": score,
            "sentiment": sent
        },
        "unique_grams": doc_uniq,
        "cleaned": ' '.join(list(row['Processed']))
    }

    table.insert_one(insert_to_cleaned)


(bowm, uniq, _) = bow(dataclean)



for u in uniq:
    insert = {
        "name": u[0],
        "idf": u[1]
    }

    table3.replace_one({ "name": u[0] }, insert, upsert=True)


matrix = tf_idf(dataclean, bowm)

size = (3,3)
rate = 0.05
iterations = 3000

som = SOM(matrix, rate, size, iterations)

curpath = os.path.abspath(os.curdir)

file_name = os.path.join(curpath, 'data\models\\training.pkl')

with open(file_name, 'wb') as f:
    pickle.dump(som, f)

insert = {
    "path": file_name,
    "iterations": iterations,
    "rate": rate,
    "size": {
        "row": size[0],
        "col": size[1],
    },
    "createdAt": datetime.today()
}

table2.insert_one(insert)


#  want to get the cluster of a tweet
#  before that, i must first have the idf values of each uniq gram of the training set
# get the uniq grams of the training set
#  get the doc grams of the training set
#  get how much documents contain this gram


# i must also have the model ready for the clustering

