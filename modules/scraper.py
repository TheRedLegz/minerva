from pprint import pprint
from modules.gram import gram_sentence
from modules.tweet_preprocessor import basic_clean, prepare_for_chunking
from modules.services import DatabaseConnection
from modules.gram import tweet_grammer, tweet_cleaner
from modules.sentiment import chunker, get_sentiment

from langdetect import detect
from datetime import datetime
import twint
import json

DEFAULT_SEARCH_STRING = '"online classes" OR "online class" OR "e-class" OR "online learning" OR "eclass" OR "face to face" OR "face-to-face" OR "lms" OR "distance learning" OR "online education" -filter:replies'


class Scraper:
    def __init__(self, search_string = DEFAULT_SEARCH_STRING):
        self.SEARCH_STRING = search_string
        self.connection = DatabaseConnection('mongodb://localhost:27017')
        self.stack = []

    def scrape(self, date = None):
        print('starting to scrape')
        c = twint.Config()
        final_date = date

        if final_date is None:
            final_date = datetime.today().strftime('%Y-%m-%d')

        c.near = 'Philippines'
        c.Search = self.SEARCH_STRING

        # non-essential parameters
        c.Until = final_date
        c.Count = True
        c.Filter_retweets = True
        c.stats = True
        # c.Store_object = True
        c.Store_json = True
        c.Output = final_date + '.json'

        twint.run.Search(c)
        # tweets = twint.output.tweets_list

        # self.save_to_db(tweets, date)

    def save_to_db(self, data, date):

        print('saving to db')

        print('TWEET COUNT', len(data))

        table = self.connection.client['minerva_raw_tweets']['rawtweets']
        table2 = self.connection.client['minerva_raw_tweets']['cleaned_tweets']
        table3 = self.connection.client['minerva_raw_tweets']['scrape_results']

        # save to raw
        # data = [a.__dict__ for a in data]

        for a in data:
            a['full_text'] = a['tweet']
            
            insert = {
                "scrape_date": date,
                "tweet_id": a['id'],
                "data": a,
                "parameters": {
                    "until": date,
                    "location": "Philippines",
                    "lang": "",
                    "query": self.SEARCH_STRING
                }
            }

            table.replace_one({ 'tweet_id': a['id'] }, insert, upsert=True)

        # save to cleaned
        print('saved raw')

        # full, cleaned, language, parameters
        cleaned = 0

        for a in data:

            try:
                lang = a['language'] if 'language' in a else detect(a['full_text'])
            except:
                continue

            if lang == 'en':
                cleaned += 1

                text = a['full_text']
                processed = basic_clean(text)
                grams = gram_sentence(processed)

                chunks = chunker(prepare_for_chunking(text))

                unq = []

                for b in grams:
                    if b not in unq:
                        unq.append(b)

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
                    "tweet_id": a['id'],
                    "full_text": text,
                    "cleaned": processed,
                    "grams": grams,
                    "unique_grams": unq,
                    "chunk_details": chunk_details,
                     "overall_sentiment": {
                        'sentiment': sent,
                        'score': scr
                    },
                    "scrape_details": {
                        "until": date,
                        "location": "Philippines",
                        "lang": "",
                        "query": self.SEARCH_STRING
                    }
                }

                table2.replace_one({ "tweet_id": a['id'] }, insert, upsert=True)
        
        results = {}
        print('saved cleaned')

        results['tweets_scraped'] = len(data)
        results['tweets_cleaned'] = cleaned
        results['scrape_date'] = date


        table3.insert_one(results)
        print('saved results')
            



        






    
        
