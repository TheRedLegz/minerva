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

# define a string of punctuation symbols
punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

l = WordNetLemmatizer()

# Functions to clean tweets


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
    return spellcheck


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


def preprocess_documents(array):
    res = []
    wordnet.ensure_loaded()
    division_n = 8
    divisions = int(len(array) / division_n)
    data = []

    for i in range(division_n):
        if i != division_n - 1:
            data.append(array[divisions*i:divisions*(i+1)])
        else:
            data.append(array[divisions*i:])

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        # Map
        for result in executor.map(_preprocess_array, data):
            futures.append(result)
        for array in futures:
            res = res + array

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return res


def _preprocess_array(array):
    res = []
    for a in array:
        preprocessed_tweet = preprocess_tweet(a)
        if len(preprocessed_tweet) == 0:
            continue
        res.append(preprocessed_tweet)

    return res


def clean_documents(array):
    res = []

    for a in array:
        res.append(basic_clean(a))

    return res
