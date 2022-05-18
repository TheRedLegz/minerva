from flask import Flask, jsonify, abort
from flask_cors import CORS
from pymongo import MongoClient
from modules.services import DatabaseConnection
from modules.tweet_preprocessor import basic_clean, preprocess_tweet
from modules.gram import gram_sentence, gram_documents, tweet_cleaner, tweet_grammer, tweet_pos
from modules.sentiment import sentimentinator
from gensim.corpora import Dictionary
from modules.vectorizer import bow, tf_idf
import numpy as np
import concurrent.futures
from modules.tweet_preprocessor import preprocess_documents
from modules.vectorizer import bag_of_words, prune_bow, tf_idf
# from modules.pca import pca
# from modules.lsi import lsi
from matplotlib import docstring, pyplot as plt
from modules.som import SOM, find_topics, get_SOM_model, print_data_to_SOM
from modules.sentiment import sentimentinator
import json
from bson import ObjectId

app = Flask(__name__)
CORS(app)

client = MongoClient('mongodb://localhost:27017')
db_raw = client['minerva_raw_tweets']
rawtweets = db_raw['rawtweets']
alltweets = rawtweets.find()
alltweets = list(alltweets[:5])


def create_tfidf():
    db_results = list(rawtweets.find())
    db_results = db_results[:5]
    data = [item['data']['full_text'] for item in db_results]

    cleaned = gram_documents(data)

    (bows, unique, docs) = bow(cleaned)
    matrix = tf_idf(cleaned, bows)

    return (matrix, unique, docs)


(weights, uniqueWords, docs) = create_tfidf()

# #----------------final.py--------------------#
# raw = client.get_raw_tweets()

# # data = [tweet['full_text'] for tweet in raw]

# # data = data[:10000]
# # csv = pd.read_csv('./data/tweets_processed.csv')

# # for tweet in csv['Content'].values[:30000]:
# #     data.append(tweet)

# data = [tweet['full_text'] for tweet in raw[:1000]]
# for i in range(500):
#     data.append(raw[i]['full_text'])

# # start_time = time.time()
# # csv = pd.read_csv('./data/tweets_processed.csv')
# # data = csv['Content'].values[:13000]
# # print("--- Execution time: %s seconds ---" % (time.time() - start_time))


# cleaned = tweet_cleaner(data)

# grammed = tweet_grammer(cleaned)
# for i, tweet_grams in enumerate(grammed):
#     grammed[i] = ' '.join(tweet_grams)
# pos_tags = tweet_pos(grammed)
# for i, tweet_grams in enumerate(pos_tags):
#     temp = []
#     for gram in tweet_grams:
#         if gram[1].startswith('NN'):
#             temp.append(gram[0])
#     grammed[i] = temp
# #------------------------------------GRAMMER---------------------------------------#
# test_data = grammed[:500]
# grammed = grammed[500:]
# (bag, unique, docs) = bow(grammed)
# #------------------------------------UNIQUE---------------------------------------#
# matrix = tf_idf(docs, bag)
# #------------------------------------TF-IDF---------------------------------------#
# lattice_size = (4, 4)
# (row, col) = lattice_size
# SOM_matrix = SOM(matrix, .3, lattice_size)
# #--------------------------------final.py----------------------------#

@app.route('/data', methods=['GET'])
def get_data():
    db_results = list(rawtweets.find()[:1000])
    data = []

    divisions = int(len(db_results) / 4)
    data_array = []
    data_array.append(db_results[0:divisions])
    data_array.append(db_results[divisions:divisions*2])
    data_array.append(db_results[divisions*2:divisions*3])
    data_array.append(db_results[divisions*3:])

    def _append_preprocessed_data(array):
        temp = []
        for a in array:
            to_add = a['data']

            to_add['preprocessed'] = preprocess_tweet(a['data']['full_text'])
            grams = gram_sentence(to_add['preprocessed'])

            to_add['unigrams'] = [token for token in grams if '_' not in token]
            to_add['bigrams'] = [
                token for token in grams if token.count('_') == 1]
            to_add['trigrams'] = [
                token for token in grams if token.count('_') == 2]
            to_add['tokens'] = to_add['unigrams'] + \
                to_add['bigrams'] + to_add['trigrams']

            temp.append(to_add)

        return temp

    with concurrent.futures.ThreadPoolExecutor() as executor:
        tempData = []
        for result in executor.map(_append_preprocessed_data, data_array):
            tempData.append(result)

        for array in tempData:
            data = data + array

    return jsonify(data)


def get_vectors():
    db_results = list(rawtweets.find())
    data = [item['data']['full_text'] for item in db_results]

    cleaned = gram_documents(data)

    (bows, unique, temp) = bow(cleaned)
    matrix = tf_idf(cleaned, bows)

    flat_matrix = np.array(matrix).flatten()

    docs = {}

    for i, doc in enumerate(cleaned):
        docs[i] = doc

    res = {
        'cols': unique,
        'tfidf': list(flat_matrix),
        'docs': docs
    }
    print(docs)
    return jsonify(res)


# get_vectors()


@app.route('/tweets', methods=['GET'])
def get_tweets():

    db_results = list(rawtweets.find())
    db_results = db_results[:5]
    data = []
    for a in db_results:
        a['data']['id'] = str(a['data']['id'])
        data.append(a['data'])

    # def get_sentiment(senti):
    #     s_data = sentimentinator(
    #         [item['full_text'] for item in senti])
    #     for i, _ in enumerate(senti):
    #         senti[i]['sentiment_score'] = s_data.iloc[i]['sentiment_score']
    #         senti[i]['sentiment'] = s_data.iloc[i]['sentiment']

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.submit(get_sentiment(data))
    return jsonify(data)


def get_tokens(res):
    cleaned = gram_documents(res)
    dt = Dictionary(cleaned)

    uni = {}
    bi = {}
    tri = {}

    cfs = dt.cfs
    dfs = dt.dfs

    res = {}

    frequencies = {}

    for word, id in dt.token2id.items():
        frequencies[word] = {}

        frequencies[word]['cfs'] = cfs[id]
        frequencies[word]['dfs'] = dfs[id]

        if '_' not in word:
            uni[word] = id
        elif word.count('_') == 1:
            bi[word] = id
        elif word.count('_') == 2:
            tri[word] = id

    res['uni'] = uni
    res['bi'] = bi
    res['tri'] = tri
    res['frequencies'] = frequencies

    return res


@app.route('/tweets/<tweet_id>')
def get_one_tweet(tweet_id):
    tweets_index = None
    temp = []
    words = []

    for index, tweets in enumerate(alltweets):
        compare = tweets['data']['id']
        
        if(str(tweets['data']['id']) == str(tweet_id)):
            res = tweets
            tweets_index = index
            break

    for item in docs[tweets_index]:
        for i, a in enumerate(uniqueWords):
            if item != uniqueWords[i][0]:
                continue
            else:
                words.append(uniqueWords[i][0])
                temp.append(weights[tweets_index][i])
                break

    res['data']['tfidf'] = temp
    res['data']['words'] = words
    res['data']['cleaned'] = basic_clean(res['data']['full_text'])
    error = jsonify({
        'error': 'Tweet does not exist'
    })
    if res:
        res['_id'] = str(res['_id'])
        return jsonify(res)

    return error


if __name__ == '__main__':
    app.run(debug=True)
