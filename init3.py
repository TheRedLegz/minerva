from modules.vectorizer import bow
from modules.services import DatabaseConnection


db = DatabaseConnection('mongodb://localhost:27017')
table = db.conn['vectors']


cleaned = list(db.get_clean_tweets())
data = [a['grams'] for a in cleaned]

(bowm, unq, _, idf) = bow(data, 4000, True)

UNQ = [a[0] for a in unq]

for idx, a in enumerate(cleaned):
    
    df = []
    doc_idf = []

    for gram in a['grams']:
        try:
            col = UNQ.index(gram)
            df.append(int(bowm[idx][col]))
            doc_idf.append(int(idf[gram]))
        except:
            continue

    insert = {
        'tweet_id': a['tweet_id'],
        'full_text': a['full_text'],
        'cleaned': a['cleaned'],
        'grams': a['grams'],
        'df': df,
        'idf': doc_idf
    }

    table.replace_one({ 'tweet_id': a['tweet_id'] }, insert, upsert=True)

