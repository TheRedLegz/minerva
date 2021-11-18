import math
import nltk
import time
import concurrent.futures
import numpy as np
from modules.tweet_preprocessor import preprocess_tweet



def bow(doc_grams):
    unique = []

    for doc in doc_grams:
        for gram in doc:
            if gram not in unique:
                unique.append(gram)



    doc_len = len(doc_grams)
    u_len = len(unique)

    bow_grams = np.zeros((doc_len, u_len), dtype=int)

    for i in range(doc_len):
        doc = doc_grams[i]

        for j in range(u_len):
            gram = unique[j]

            count = doc.count(gram)

            bow_grams[i][j] = count

    return (bow_grams, unique)

# TODO: Implement multiprocessing
def _bag_of_words_sub_process(array, to_preprocess):
    doc_grams = []

    for doc in array:
        string = ''

        if to_preprocess:
            string = preprocess_tweet(doc)
        else:
            string = doc
        
        if len(string) == 0:
            continue

        unigrams = nltk.word_tokenize(string)
        unigrams = [u for u in unigrams if len(u) > 2]
        bigrams = nltk.bigrams(unigrams)
        bigrams = map(lambda x: x[0] + '_' + x[1], bigrams)
        uni_bi_grams = list(bigrams) + unigrams
        doc_grams.append(list(uni_bi_grams))

    return doc_grams

def bag_of_words(document_array, to_preprocess=True):
    start_time = time.time()
    doc_grams = []

    for doc in document_array:
        string = ''

        if to_preprocess:
            string = preprocess_tweet(doc)
        else:
            string = doc
        
        # Consider just not dropping empty documents
        if len(string) == 0:
            continue

        unigrams = nltk.word_tokenize(string)
        unigrams = [u for u in unigrams if len(u) > 2]
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
        
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))
    return (bow_grams, unique_grams, doc_grams)
    
def prune_bow(bow, tf_idf_threshold = 1):
    (bow_grams, unique, docs) = bow


    original = np.copy(bow_grams)
    total_docs = len(docs)

    start_time = time.time()

    tres = []
    division_n = 8
    divisions = int(len(original[0]) / division_n)
    data = [] 

    #----- MULTIPROCESSING ------#
    # for i in range(division_n):
    #     copy_bow = np.copy(original)
    #     if i != division_n - 1:
    #         data.append(copy_bow[:,divisions*i:divisions*(i+1)])
    #     else:
    #         data.append(copy_bow[:,divisions*i:])

    # #Figure out what to do here
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     for result in executor.map(_prune_bow_sub_method, data):
    #         tres.append(result)

    # for i, array in enumerate(tres):
    #     if i == 0:
    #         data = tres[i]
    #     else:
    #         np.hstack(data,array)
    # print(data.shape)


    # Get Document Frequency == docs with gram / doc #
    for i in range(len(unique)-1, -1, -1):
        # loop_time = time.time()
        df =  np.count_nonzero(original[:, i:i+1]) / total_docs

        if df < tf_idf_threshold/len(docs):
            unique.pop(i)
            bow_grams = np.delete(bow_grams, i, 1)
        # print("--- Loop time: %s seconds ---" % (time.time() - loop_time))
            
    for i in range(len(docs)):
        docs[i] = [word for word in docs[i] if word in unique]
        
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return (bow_grams, unique, docs)

def _prune_bow_sub_method(bow, tf_idf_threshold, sub_array):
    (bow_grams, unique, docs) = bow
    total_docs = len(docs)
    for i in range(len(unique)-1, -1, -1):
        # loop_time = time.time()
        df =  np.count_nonzero(sub_array[:, i:i+1]) / total_docs

        if df < tf_idf_threshold/len(docs):
            unique.pop(i)
            bow_grams = np.delete(bow_grams, i, 1)
        # print("--- Loop time: %s seconds ---" % (time.time() - loop_time))


def tf_idf(document_array, bow = None):

    if bow is None:
        (bow, x, y) = bag_of_words(document_array)

    matrix = np.zeros(bow.shape, dtype=float)
    doc_count = len(document_array)

    for i, row in enumerate(bow):
        word_count = np.sum(row)

        for j, col in enumerate(row):
            
            # NOTE you might want to remove the document instead but not now

            if word_count == 0:
                matrix[i][j] = 0
                continue
                
            tf = col / word_count
            
            has_word_count = np.count_nonzero(bow[:, j:j+1])
            idf = math.log((doc_count / (has_word_count)))

            matrix[i][j] = tf * idf 
    return matrix
