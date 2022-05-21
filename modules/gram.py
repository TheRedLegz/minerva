from gensim.models.phrases import Phrases, Phraser
from gensim.utils import tokenize
from gensim.parsing.preprocessing import STOPWORDS
import modules.tweet_preprocessor as p
import pandas as pd
from pprint import pprint as print
import os.path
import concurrent.futures
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

add = [
    'online',
    'class',
    'learning',
    'distance',
    'education',
    'face',
    'lms'
]

STOPWORDS = list(STOPWORDS) + add

my_path = os.path.abspath(os.path.dirname(__file__))
bpath = os.path.join(my_path, "../data/models/bigram_model.pkl")
tpath = os.path.join(my_path, "../data/models/trigram_model.pkl")

try:

    bigram_phrases = Phrases.load(bpath)
    trigram_phrases = Phrases.load(tpath)

except:
    print('Making a Phrases model')

    path = os.path.join(my_path, "../data/test_tweets.csv")

    test_tweets = pd.read_csv(path)
    test_data = test_tweets['Tweet']

    clean_test_data = p.clean_documents(test_data)
    tokenized_data = [doc.split(' ') for doc in clean_test_data]

    bigram_phrases = Phrases(tokenized_data, min_count=1, threshold=50)
    trigram_phrases = Phrases(
        bigram_phrases[tokenized_data], min_count=3, threshold=10)

    bigram_phrases.save(bpath)
    trigram_phrases.save(tpath)


bigram_model = Phraser(bigram_phrases)
trigram_model = Phraser(trigram_phrases)


def clean_token(word):
    if '_' in word:
        split = word.split('_')

        if len(split[0]) > 2 and len(split[1]) > 2 and split[1] not in STOPWORDS and split[0] not in STOPWORDS:
            return p.lemmatize(split[0]) + '_' + p.lemmatize(split[1])

    elif len(word) > 2 and word not in STOPWORDS:
        return p.lemmatize(word)


def clean_document_tokens(doc):
    """
    Remove bigrams/trigrams that contain a stopword
    Also, lemmatizes valid bigrams/trigrams

    Example:

    is_playing => removed
    stupid_cats => stupid_cat

    Input:
    Array of bigrams/trigrams

    """

    res = []

    for word in doc:
        cleaned = clean_token(word)

        if cleaned and cleaned not in res:
            res.append(cleaned)

    return res


def gram_sentence(data, clean=True):

    if clean:
        data = p.basic_clean(data)

    if not isinstance(data, list):
        data = data.split(' ')

    results = trigram_model[data]
    results = clean_document_tokens(results)

    return results


def gram_documents(data):
    results = [gram_sentence(doc) for doc in data]
    return results


def tweet_pos(obj):
    res = []

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        for results in executor.map(p.pos_tag, obj):
            res.append(results)

    return res


def tweet_cleaner(obj):
    res = []

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        for results in executor.map(p.basic_clean, obj):
            res.append(results)

    return res


def tweet_grammer(docs):
    split = [s.split(' ') for s in docs]
    res = trigram_model[split]

    grammed_docs = []
    wordnet.ensure_loaded()

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        for result in executor.map(clean_document_tokens, res):
            grammed_docs.append(result)

    return grammed_docs
