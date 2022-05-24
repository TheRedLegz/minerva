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
    valid_grams = []

    for gram in a['unique_grams']:
        try:
            col = UNQ.index(gram)

            insertdf = int(bowm[idx][col])
            insertidf = int(idf[gram])

            df.append(insertdf)
            doc_idf.append(insertidf)
            valid_grams.append(gram)
        except:
            continue

    insert = {
        'tweet_id': a['tweet_id'],
        'full_text': a['full_text'],
        'cleaned': a['cleaned'],
        'grams': valid_grams,
        'word_count': len(a['grams']),
        'df': df,
        'idf': doc_idf
    }

    table.replace_one({ 'tweet_id': a['tweet_id'] }, insert, upsert=True)

