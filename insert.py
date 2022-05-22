from pymongo import MongoClient
import json
from pprint import pprint as print

client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']


jsonpath = './data/scraped.json'

file = open(jsonpath, encoding="utf-8")


data = json.load(file)
date = "2021-11-20"


count = 0
for a in data:
    a['full_text'] = a['tweet']

    count = count + 1
    insert = {
        "scrape_date": date,
        "tweet_id": a['id'],
        "parameters": {
            "until": date,
            "location": "Philippines",
            "lang": "",
            "query": '"online classes" OR "online class" OR "e-class" OR "online learning" OR "eclass" OR "face to face" OR "face-to-face" OR "lms" OR "distance learning" OR "online education" -filter:replies'
        },
        "data": a
    }


    print(count)
    rawtweets.insert_one(insert)


    