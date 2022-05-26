

from modules.gram import gram_sentence
from modules.tweet_preprocessor import basic_clean
from modules.services import DatabaseConnection
from modules.gram import tweet_grammer, tweet_cleaner
from modules.sentiment import chunker, get_sentiment

from langdetect import detect
from datetime import datetime
import twint
import json

DEFAULT_SEARCH_STRING = '"online classes" OR "online class" OR "e-class" OR "online learning" OR "eclass" OR "face to face" OR "face-to-face" OR "lms" OR "distance learning" OR "online education" OR "education" OR "class" -filter:replies'


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
            date = datetime.today().strftime('%Y-%m-%d')

        c.near = 'Philippines'
        c.Search = self.SEARCH_STRING

        # non-essential parameters
        c.Until = date
        c.Count = True
        c.Filter_retweets = True
        c.stats = True
        c.Store_object = True

        twint.run.Search(c)
        tweets = twint.output.tweets_list

        self.save_to_db(tweets, date)

    def save_to_db(self, data, date):

        print('saving to db')

        table = self.connection['minerva_raw_tweets']['rawtweets']
        table2 = self.connection['minerva_raw_tweets']['cleaned_tweets']
        table3 = self.connection['minerva_raw_tweets']['scrape_results']

        # save to raw

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
            lang = a['data']['language']

            if not lang:
                lang = detect(a['data']['full_text'])


            if lang == 'en':
                cleaned += 1

                text = a['data']['full_text']
                processed = basic_clean(text)
                grams = gram_sentence(processed)

                chunks = chunker(processed)

                chunk_details = []

                for chunk in chunks:
                    res = {}
                    score = get_sentiment(chunk)
                    res['score'] = score
                    res['chunk'] = chunk
                    
                    chunk_details.append(res)


                insert = {
                    "tweet_id": a['data']['id'],
                    "full_text": text,
                    "cleaned": processed,
                    "grams": grams,
                    "chunk_details": chunk_details,
                    "overall_sentiment": get_sentiment(text),
                    "scrape_details": a['parameters']
                }

                table2.replace_one({ "tweet_id": a['data']['id'] }, insert, upsert=True)
        
        results = {}
        print('saved cleaned')

        results['tweets_scraped'] = len(data)
        results['tweets_cleaned'] = len(cleaned)
        results['scrape_date'] = date


        table3.insert_one(results)
        print('saved results')


            



        






    
        
