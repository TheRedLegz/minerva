import math
import numpy as np
from pprint import pprint
import nltk
import re
import string as str
from nltk.stem import WordNetLemmatizer
from numpy.lib import unique

nltk.download('stopwords')
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(string):
    
    def remove_numbers(string):
        res = re.sub(r'\d+', '', string)
        return res


    def remove_punctation(string):
        res = string.translate(str.maketrans("", ""), str.punctuation)
        return res

    def remove_stop_words(string):
        res = [w for w in string if w not in stop_words]
        return res


    string = string.lower().strip()
    string = remove_numbers(string)
    string = remove_punctation(string)
    string = remove_stop_words(string)
    string = lemmatizer.lemmatize(string)

    return string


def bag_of_words(document_array):


    document_bigrams = []

    for doc in document_array:
        tokenized = nltk.word_tokenize(doc)
        grammed = nltk.bigrams(tokenized)
        grammed = map(lambda x: x[0] + '_' + x[1], grammed)
        document_bigrams.append(list(grammed))


    unique_bigrams = []

    for doc in document_bigrams:
        for bigram in doc:
            if bigram not in unique_bigrams:
                unique_bigrams.append(bigram)


    # doc = row, bigram = col
    document_count = len(document_bigrams)
    bigram_count = len(unique_bigrams)

    bow_bigrams = np.zeros((document_count, bigram_count), dtype=int)
    

    for i in range(document_count):
        doc = document_bigrams[i]

        for j in range(bigram_count):
            bigram = unique_bigrams[j]
            count = doc.count(bigram)

            bow_bigrams[i][j] = int(count)

    
    return (bow_bigrams, unique_bigrams)


def tf_idf(document_array, bow = None):

    if bow is None:
        (bow, _) = bag_of_words(document_array)

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


tf_idf(data)