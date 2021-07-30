import nltk

nltk.download('stopwords')
nltk.download('wordnet')

import math
import numpy as np
from pprint import pprint
import re
import string as str
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(string):
    
    def remove_numbers(string):
        res = re.sub(r'\d+', '', string)
        return res

    def remove_punctation(string):
        res = string.translate(string.maketrans("", "", str.punctuation))
        return res

    # NOTE this returns an array
    def remove_stop_words(string):
        split = string.split(' ')
        res = [w for w in split if w not in stop_words]
        return res

    def lemmatize_string(string):
        split = string

        if not isinstance(string, list):    
            split = string.split(' ')

        res = []

        for word in split:
            lemma = lemmatizer.lemmatize(word)
            res.append(lemma)

        return ' '.join(res)

    string = string.lower().strip()
    string = remove_numbers(string)
    string = remove_punctation(string)
    string = remove_stop_words(string)
    string = lemmatize_string(string)

    return string

def bag_of_words(document_array):

    doc_grams = []

    for doc in document_array:
        string = preprocess(doc)
        unigrams = nltk.word_tokenize(string)
        bigrams = nltk.bigrams(unigrams)
        bigrams = map(lambda x: x[0] + '_' + x[1], bigrams)
        uni_bi_grams = list(bigrams) + unigrams
        doc_grams.append(list(uni_bi_grams))


    unique_grams = []

    for doc in doc_grams:
        for bigram in doc:
            if bigram not in unique_grams:
                unique_grams.append(bigram)
    

    # doc = row, bigram = col
    document_count = len(doc_grams)
    gram_count = len(unique_grams)

    bow_grams = np.zeros((document_count, gram_count), dtype=int)
    

    for i in range(document_count):
        doc = doc_grams[i]

        for j in range(gram_count):
            gram = unique_grams[j]
            count = doc.count(gram)

            bow_grams[i][j] = int(count)

    return (bow_grams, unique_grams, doc_grams)


def tf_idf(document_array, bow = None):

    if bow is None:
        (bow, x, y) = bag_of_words(document_array)

    matrix = np.zeros(bow.shape, dtype=float)
    doc_count = len(document_array)

    for i, row in enumerate(bow):

        word_count = np.sum(row)

        for j, col in enumerate(row):
            
            tf = col / word_count
        
            has_word_count = np.count_nonzero(bow[:, j:j+1])
            idf = math.log(doc_count / (has_word_count + 1))

            matrix[i][j] = tf * idf

    return matrix



# raw = pd.read_json('sample.json')
# data = []

# for i, row in raw.iterrows():
#     data.append(row['full_text'])


data = [
    'this is a sentence',
    'this is not a sentence',
    'sentence something',
    'something cat cute',
    'cat sentence',
]


pprint(tf_idf(data))