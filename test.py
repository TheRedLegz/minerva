from modules.gram import gram_documents
from modules.services import DatabaseConnection
from pprint import pprint
import re

conn = DatabaseConnection('mongodb://localhost:27017')

cleaned_tweets = conn.get_clean_tweets()
cleaned_tweets = [clean["cleaned"] for clean in cleaned_tweets]
grams = gram_documents(cleaned_tweets)
count = 0
total_count = 0
for doc in grams:
    for gram in doc:
        if (re.search('\w+_\w+', gram)):
            count += 1
        total_count += 1
        


pprint(grams)
print(count)
print(total_count)