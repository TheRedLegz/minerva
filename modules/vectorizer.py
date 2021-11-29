import math
import nltk
import time
import concurrent.futures
import numpy as np
from modules.tweet_preprocessor import preprocess_tweet
from gensim.corpora import Dictionary
from pprint import pprint as print

def bow(doc_grams, max = 5000):
    unique = {}

    for doc in doc_grams:
        for gram in doc:
            if gram not in unique:
                unique[gram] = 0
            else:
                unique[gram] = unique[gram] + 1
    

    unique = list(sorted(unique.items(), key=lambda item: item[1], reverse=True))[:5000]

    unique = [val[0] for val in unique]

    doc_len = len(doc_grams)
    u_len = len(unique)

    
    # TODO sparse matrix representation

    bow_grams = np.zeros((doc_len, u_len), dtype=int)

    for i in range(doc_len - 1, -1, -1):


        doc = doc_grams[i]

        for j in range(u_len):
            gram = unique[j]

            count = doc.count(gram)

            bow_grams[i][j] = count

        if np.count_nonzero(bow_grams[i]) == 0:
            del doc_grams[i]

    return (bow_grams, unique, doc_grams)







def bag_of_words(preprocessed_tweets, process_count = 4):
    start_time = time.time()
    doc_grams = []

    # TODO: Add multi processing here
    with concurrent.futures.ThreadPoolExecutor(process_count) as executor:
        for result in executor.map(_bag_of_words_sub_method, preprocessed_tweets):
            # NOTE: Removes empty docs
            if result != '':
                doc_grams.append(result)

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
        
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))
    return (bow_grams, unique_grams, doc_grams)
    
# TODO: For ThreadPool
def _bag_of_words_sub_method(doc):
    string = doc
    
    if len(string) == 0:
        return ''

    unigrams = nltk.word_tokenize(string)
    unigrams = [u for u in unigrams if len(u) > 2]
    bigrams = nltk.bigrams(unigrams)
    bigrams = map(lambda x: x[0] + '_' + x[1], bigrams)
    uni_bi_grams = list(bigrams) + unigrams
    return list(uni_bi_grams)

def prune_bow(bow, tf_idf_threshold = 1, thread_count = 4):
    start_time = time.time()
    (bow_grams, unique_grams, docs) = bow
    # TODO: Add removal of keywords in docgrams
    total_docs = len(docs)

    original = np.copy(bow_grams)
    original_t = np.transpose(original)

    data = [] 
    pruned_grams = []

    for i, gram in enumerate(original_t):
        data.append((unique_grams[i], total_docs, tf_idf_threshold, gram))

    # NOTE: Multiprocessing application
    with concurrent.futures.ThreadPoolExecutor(thread_count) as executor:
        for result in executor.map(_prune_bow_sub_method, data):
            (_, gram, _) = result
            # Filter out pruned grams
            if len(gram) > 0:
                pruned_grams.append(result)
    
    # NOTE: Serial Method
    # for data_item in data:
    #     result = _prune_bow_sub_method(data_item)
    #     # Filter out pruned grams
    #     if len(result) > 0:
    #         pruned_grams.append(result)

    # Transfers unique keywords into list
    unique_grams = [unique for (unique, _, _) in pruned_grams]
    # NOTE: Find a way to optimize this to remove the number of loops?
    # Removes pruned unique words from doc_grams
    for i in range(len(docs)):
        docs[i] = [word for word in docs[i] if word in unique_grams]
    # Pairs the unique gram with its idf score
    unique_grams = [(unique, idf) for (unique, _, idf) in pruned_grams]
    # Transfers grams into list
    pruned_grams = [gram for (_, gram, _) in pruned_grams]
    # From gram x doc -> doc x gram
    pruned_grams = np.transpose(pruned_grams)
        
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return (pruned_grams, unique_grams, docs)

# TODO: Change this to remove loops
def _prune_bow_sub_method(data):
    (unique, total_docs, tf_idf_threshold, gram) = data
    total_keyword_count = np.count_nonzero(gram)
    df =  np.count_nonzero(gram) / total_docs
    idf = total_docs / total_keyword_count

    if df < tf_idf_threshold/total_docs:
        return (unique, [], -1)
    
    return (unique, gram, idf)

def tf_idf(document_array, bow = None):
    start_time = time.time()

    if bow is None:
        (bow, x, y) = bag_of_words(document_array)

    matrix = np.zeros(bow.shape, dtype=float)
    doc_count = len(document_array)

    col_sums = (bow != 0).sum(0)

    data = []
    matrix = []

    for i, row in enumerate(bow):

        data.append((row, doc_count, col_sums))
        
        # word_count = np.sum(row)
        # for j, col in enumerate(row):
            
        #     if word_count == 0:
        #         matrix[i][j] = 0
        #         continue
                
        #     tf = col / word_count
            
        #     has_word_count = np.count_nonzero(bow[:, j:j+1])
        #     idf = math.log((doc_count / (has_word_count)))

        #     matrix[i][j] = tf * idf 

    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        for _ in executor.map(_tf_idf_sub, data):
            matrix.append(_)

    return np.asarray(matrix) 


def _tf_idf_sub(data):
    (row, doc_count, col_sums) = data

    res = np.zeros(len(row), dtype=float)

    word_count = np.sum(row)

    for j, col in enumerate(row):
        if word_count == 0:
            res[j] = 0
            continue
        
        tf = col / word_count
        has_word_count = col_sums[j]

        idf = math.log((doc_count / (has_word_count)))

        res[j] = tf * idf 
        
    return res