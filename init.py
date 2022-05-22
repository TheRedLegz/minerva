from modules.gram import gram_sentence
from modules.tweet_preprocessor import basic_clean
from modules.services import DatabaseConnection
from modules.gram import tweet_grammer, tweet_cleaner
from modules.sentiment import chunker, get_sentiment
from langdetect import detect

db = DatabaseConnection('mongodb://localhost:27017')


raw = list(db.get_full_raw_tweets())

table = db.conn['cleaned_tweets']

for a in raw:
    lang = a['data']['language']

    if not lang:
        lang = detect(a['data']['full_text'])


    if lang == 'en':
        text = a['data']['full_text']
        processed = basic_clean(text)
        grams = gram_sentence(processed)

        chunks = chunker(processed)

        chunk_details = []

        for chunk in chunks:
            res = {}
            (sentiment, score) = get_sentiment(chunk)
            res['score'] = score
            res['sentiment'] = sentiment
            res['chunk'] = chunk
            
            chunk_details.append(res)

        (sent, scr) = get_sentiment(text)

        insert = {
            "tweet_id": a['data']['id'],
            "full_text": text,
            "cleaned": processed,
            "grams": grams,
            "chunk_details": chunk_details,
            "overall_sentiment": {
                'sentiment': sent,
                'score': scr
            },
            "scrape_details": a['parameters']
        }

        table.replace_one({ 'tweet_id': a['data']['id'] }, insert, upsert=True)

    res = {}





