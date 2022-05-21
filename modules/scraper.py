

from minerva.modules.gram import gram_sentence
from minerva.modules.tweet_preprocessor import basic_clean
from modules.services import DatabaseConnection
from modules.gram import tweet_grammer, tweet_cleaner
from modules.sentiment import chunker, get_sentiment

from langdetect import detect
import twint
import json

DEFAULT_SEARCH_STRING = '"online classes" OR "online class" OR "e-class" OR "online learning" OR "eclass" OR "face to face" OR "face-to-face" OR "lms" OR "distance learning" OR "online education" -filter:replies'


class Scraper:
    def __init__(self, search_string = DEFAULT_SEARCH_STRING):
        self.SEARCH_STRING = search_string
        self.connection = DatabaseConnection('mongodb://localhost:27017')

    def scrape(self, date):
        c = twint.Config()
        
        c.near = 'Philippines'
        c.Until = date
        c.Count = True
        c.Filter_retweets = True
        c.stats = True
        c.Search = self.SEARCH_STRING
        c.Store_json = True
        c.Output = date + '.json'

        twint.run.Search(c)

    def save_to_db(self, date, path):
        file = open(path, encoding="utf-8")
        data = json.load(file)

        table = self.connection['minerva_raw_tweets']['rawtweets']
        table2 = self.connection['minerva_raw_tweets']['cleaned_tweets']

        # save to raw

        for a in data:
            a['full_text'] = a['tweet']
            a['parameters'] = {
                "until": date,
                "location": "Philippines",
                "lang": "",
                "query": self.SEARCH_STRING
            }

            count = count + 1
            insert = {
                "scrape_date": date,
                "tweet_id": a['id'],
                "data": a
            }

            table.insert_one(insert)


        # save to cleaned

        # full, cleaned, language, parameters

        for a in data:
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

                table2.insert_one(insert)


            
            



        






    
        
