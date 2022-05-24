import math
import nltk
import time
import concurrent.futures
import numpy as np
from modules.tweet_preprocessor import preprocess_tweet
from gensim.corpora import Dictionary


def generate_dbvector_matrix(data, uniq):
    size = (len(data), len(uniq))

    matrix = np.zeros(size)

    for i, row in enumerate(data):
        mrow = matrix[i]
        df = row['df']
        idf = row['idf']
        wordcount = row['word_count']
        doccount = len(data)

        print('len: ', len(df))
        print('grams: ', len(row['grams']))

        for k, gram in enumerate(row['grams']):

            idx = uniq.index(gram)
            
            score1 = df[k] / wordcount
            score2 = math.log((doccount + 1)/(idf[k] + 1)) + 1

            mrow[idx] = score1 * score2

    return matrix



def bow(doc_grams, max=4000, getIdf = False):
    unique = {}

    for doc in doc_grams:
        for gram in doc:
            if gram not in unique:
                unique[gram] = 1
            else:
                unique[gram] = unique[gram] + 1

    
    unique = list(
        sorted(unique.items(), key=lambda item: item[1], reverse=True))[:max]

    unique = [val[0] for val in unique]

    doc_len = len(doc_grams)
    u_len = len(unique)

    # TODO sparse matrix representation

    bow_grams = np.zeros((doc_len, u_len), dtype=int)

    idf = {}

    for i in range(doc_len - 1, -1, -1):

        doc = doc_grams[i]

        for j in range(u_len):
            gram = unique[j]
            count = doc.count(gram)
                    
            bow_grams[i][j] = count

        if np.count_nonzero(bow_grams[i]) == 0:
            del doc_grams[i]
            continue
        
        temp = []
        for word in doc:
            if word not in temp:
                temp.append(word)

                if word not in idf:
                    idf[word] = 1
                else:
                    idf[word] += 1


    transposed_bow = np.transpose(np.copy(bow_grams))
    for i, gram in enumerate(transposed_bow):
        unique[i] = (unique[i], math.log(
            (doc_len + 1) / (np.count_nonzero(gram) + 1)) + 1)
    
    if getIdf:
        return (bow_grams, unique, doc_grams, idf)
    
    return (bow_grams, unique, doc_grams)

def euc_distance(input, num_features):
    sum = 0

    for n in range(num_features):
        sum = sum + pow(input[n], 2)

    return math.sqrt(sum)

def tf_idf(document_array, bow=None):
    start_time = time.time()

    if bow is None:
        (bow, x, y) = bow(document_array)

    matrix = np.zeros(bow.shape, dtype=float)
    doc_count = len(document_array)

    col_sums = np.count_nonzero(bow, 0)

    data = []
    matrix = []

    for i, row in enumerate(bow):

        data.append((row, doc_count, col_sums))

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        for _ in executor.map(_tf_idf_sub, data):
            matrix.append(_)

    return np.asarray(matrix)


def _tf_idf_sub(data):
    (row, doc_count, col_sums) = data
    total_distance = 0
    res = np.zeros(len(row), dtype=float)

    word_count = np.sum(row)

    for j, col in enumerate(row):
        if word_count == 0:
            res[j] = 0
            continue

        tf = col / word_count
        has_word_count = col_sums[j]

        idf = math.log(((doc_count + 1) / (has_word_count + 1))) + 1

        res[j] = tf * idf

    total_distance = euc_distance(res, len(res))

    for i in range(len(row)):
        if total_distance == 0:
            res[i] = 0
        else:
            res[i] /= total_distance

    return res
