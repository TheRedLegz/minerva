from modules.gram import gram_documents
import json
from pprint import pprint as print
from gensim.corpora import Dictionary

file = open('data/scraped.json',encoding="utf8")
data = json.load(file)
data = [doc['tweet'] for doc in data]

tokens = gram_documents(data)



def get_token_count(dict):
    keys = dict.token2id
    cfs = dict.cfs

    result = {}
    
    for key in cfs:
        token_name = list(keys.keys())[list(keys.values()).index(key)]
        result[token_name] = cfs[key]

    return result

dictionary = Dictionary(tokens)

testtt = dictionary.token2id.keys()


unique = []

for bigram in testtt:
    if bigram not in unique and '_' in bigram:
        unique.append(bigram)

# trigrams and bigrams
print(unique)


# all tokens
# print(sorted(get_token_count(dictionary).items(), key=lambda x: x[1], reverse=True)[:150])