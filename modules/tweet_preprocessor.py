import pandas as pd
import re
import gensim
import nltk
import time
import concurrent.futures
from googletrans import Translator
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.preprocessing import STOPWORDS
import os.path
# define a string of punctuation symbols
punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

l = WordNetLemmatizer()

# Functions to clean tweets


def clean_token(word):
    if '_' in word:
        split = word.split('_')

        if len(split[0]) > 2 and len(split[1]) > 2 and split[1] not in STOPWORDS and split[0] not in STOPWORDS:
            return lemmatize(split[0]) + '_' + lemmatize(split[1])

    elif len(word) > 2 and word not in STOPWORDS:
        return word


def count_english(tweet_array):
    translator = Translator()
    res = []
    for tweet in tweet_array:
        langs = translator.detect(tweet)
        if langs.lang == 'en':
            res.append(tweet)
    translations = translator.translate(res)
    return translations


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

        if cleaned:
            res.append(cleaned)

    return res


def gram_sentence(data):
    data = basic_clean(data)
    data = data.split(' ')

    results = trigram_model[data]
    results = clean_document_tokens(results)

    return results


def remove_links(tweet):
    """Takes a string and removes web links from it"""
    tweet = re.sub(r'http\S+', '', tweet)   # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')   # remove [links]
    tweet = re.sub(r'pic.twitter\S+', '', tweet)
    return tweet


def spell_check(string):
    spellcheck = TextBlob(string)
    spellcheck = spellcheck.correct()
    return str(spellcheck)


def translateinator(string):
    translator = Translator()
    res = translator.translate(string)
    return res.text


def remove_users(tweet):
    """Takes a string and removes retweet and @user information"""
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)',
                   '', tweet)  # remove re-tweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)',
                   '', tweet)  # remove tweeted at
    return tweet


def remove_hashtags(tweet):
    """Takes a string and removes any hash tags"""
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
    return tweet


def remove_av(tweet):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
    return tweet


def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS \
                and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemmatize(token))
    return result


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(token):
    """Returns lemmatization of a token"""
    return l.lemmatize(token, get_wordnet_pos(token))


def remove_non_ascii(tweet):
    return tweet.encode("ascii", "ignore").decode()


def remove_html_tags(string):
    res = re.sub(
        r'&(gt|lt|amp|nbsp|quot|apos|cent|pound|yen|euro|copy|reg);', '', string)
    return res


def preprocess_tweet(tweet):
    """Main master function to clean tweets, stripping noisy characters, and tokenizing use lemmatization"""
    tweet = translateinator(tweet)
    tweet = spell_check(tweet)
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_av(tweet)
    tweet = remove_html_tags(tweet)
    tweet = tweet.lower()  # lower case
    tweet = remove_non_ascii(tweet)
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet_token_list = tokenize(tweet)  # apply lemmatization and tokenization
    tweet = ' '.join(tweet_token_list)
    return tweet


def basic_clean(tweet):
    """Main master function to clean tweets only without tokenization or removal of stopwords"""
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_av(tweet)
    tweet = remove_html_tags(tweet)
    tweet = remove_non_ascii(tweet)
    tweet = tweet.lower()  # lower case
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet = re.sub('📝 …', '', tweet)
    return tweet

# TODO:
#   - Add a way to check if a tweet preexists in preprocessed tweet database


def preprocess_documents(raw_tweets, thread_count=8):
    start_time = time.time()

    res = []
    wordnet.ensure_loaded()

    with concurrent.futures.ThreadPoolExecutor(thread_count) as executor:
        # Map
        for result in executor.map(_preprocess_array, raw_tweets):
            res.append(result)
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return res


def _preprocess_array(document):
    tweet_id = document['tweet_id']
    preprocessed_text = preprocess_tweet(document['full_text'])
    print(preprocessed_text)

    return {'tweet_id': tweet_id, 'preprocessed_text': preprocessed_text}


def clean_documents(array):
    res = []

    for a in array:
        res.append(basic_clean(a))

    return res


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

    clean_test_data = clean_documents(test_data)
    tokenized_data = [doc.split(' ') for doc in clean_test_data]

    bigram_phrases = Phrases(tokenized_data, min_count=1, threshold=50)
    trigram_phrases = Phrases(
        bigram_phrases[tokenized_data], min_count=3, threshold=10)

    bigram_phrases.save(bpath)
    trigram_phrases.save(tpath)


bigram_model = Phraser(bigram_phrases)
trigram_model = Phraser(trigram_phrases)
