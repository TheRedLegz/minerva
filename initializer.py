from datetime import datetime
from modules.som import tweet_find_cluster
from modules.gram import tweet_grammer
from modules.sentiment import chunker
from modules.services import DatabaseConnection
from modules.sentiment import get_sentiment, chunker
from modules.tweet_preprocessor import prepare_for_chunking
from modules.vectorizer import bow, tf_idf
from modules.som import SOM
from modules.lib import init_save_clean,filter_non_eng

import json
import os

import pickle
import codecs
import pandas as pd

"""

INITIALIZATION

Step 1

Make the SOM model
- [] Load the csv
- [] Take the preprocessed text (already grammed)
- [] Save its features and idf
- [] Make the bow
- [] Make the tfidf
- [] Make the SOM
- [] Save the SOM


"""
db = DatabaseConnection('mongodb://localhost:27017')
table = db.conn['training_clean']
table2 = db.conn['training_model']
table3 = db.conn['training_features']


df = pd.read_csv('./data/tweets_processed.csv')
datarows = list(df.iterrows())[:200]

dataclean = [json.loads(a.replace("'", "\"")) for a in list(df['Processed'])[:200]]

for a in dataclean:
    a = [b for b in a if 'learn' not in b and 'education' not in b]

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

    grams = json.loads(row['Processed'].replace("'", "\""))
    grams = [g for g in grams if 'learn' not in g and 'education' not in g]

    for a in grams:
        if a not in doc_uniq:
            doc_uniq.append(a)
        

    insert_to_cleaned = {
        "grams": grams,
        "full_text": full_text,
        "chunk_details": chunk_details,
        "overall_sentiment": {
            "score": score,
            "sentiment": sent
        },
        "unique_grams": doc_uniq,
        "cleaned": ' '.join(grams)
    }

    table.insert_one(insert_to_cleaned)


(bowm, uniq, _) = bow(dataclean)
unq_list = []

for u in uniq:
    insert = {
        "name": u[0],
        "idf": u[1]
    }

    unq_list.append(u[0])
    table3.replace_one({ "name": u[0] }, insert, upsert=True)


matrix = tf_idf(dataclean, bowm)

# make topic coherence

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

# 

raw = db.get_full_raw_tweets()
filt = filter_non_eng(raw)

init_save_clean(filt)