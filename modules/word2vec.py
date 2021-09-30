from nltk import word_tokenize
from modules.tweet_preprocessor import preprocess_documents

from gensim.models import Word2Vec

from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']

#Add word2vec parameters here
def get_word2vec_from_data(data, to_preprocess=True):
    
    if to_preprocess:
        data = preprocess_documents(data)
        
    for i in range(len(data)):
        data[i] = word_tokenize(data[i])

    model = Word2Vec(data, workers=1, min_count=2, vector_size=20)
    
    return model




# if __name__ == "__main__":
#     db_results = list(rawtweets.find())
#     data = []   

#     print("Started Loading Data")
#     # Assigning tweets in variable
#     for a in db_results:
#         data.append(a['data']['full_text'])
#     print("Finished Loading Data")

#     print("Started Preprocessing")
#     data = preprocess_documents(data)
#     print("Finished Preprocessing")

#     for i in range(len(data)):
#         data[i] = word_tokenize(data[i])

#     model = Word2Vec(data, workers=1, min_count=2, vector_size=20)
#     print(model.wv.key_to_index)
#     # print(len(model.wv['connection']))