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

def bag_of_words(preprocessed_tweets, to_preprocess=False):
    start_time = time.time()
    doc_grams = []

    # TODO: Add multi processing here
    for doc in preprocessed_tweets:
        string = ''

        if to_preprocess:
            string = preprocess_tweet(doc)
        else:
            string = doc['preprocessed_text']
        
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
    
def prune_bow(bow, tf_idf_threshold = 1, thread_count = 8):
    (bow_grams, unique_grams, docs) = bow
    total_docs = len(docs)

    start_time = time.time()
    original = np.copy(bow_grams)
    original_t = np.transpose(original)

    data = [] 
    pruned_grams = []

    for i, gram in enumerate(original_t):
        data.append((unique_grams[i], docs, total_docs, tf_idf_threshold, gram))

    # with concurrent.futures.ThreadPoolExecutor(thread_count) as executor:
    #     for result in executor.map(_prune_bow_sub_method, data):
    #         pruned_grams.append(result)

    for data_item in data:
        pruned_grams.append(_prune_bow_sub_method(data_item))


    # Filters out pruned grams
    pruned_grams = [(unique, gram) for (unique, gram) in pruned_grams if len(gram) > 0]
    # Transfers unique keywords into list
    unique_grams = [unique for (unique, _) in pruned_grams]
    # Transfers grams into list
    pruned_grams = [gram for (_, gram) in pruned_grams]
    # From gram x doc -> doc x gram
    pruned_grams = np.transpose(pruned_grams)
    # Removes pruned unique words from doc_grams
    for i in range(len(docs)):
        docs[i] = [word for word in docs[i] if word in unique_grams]
        
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return (pruned_grams, unique_grams, docs)

# TODO: Change this to remove loops
def _prune_bow_sub_method(data):
    (unique, docs, total_docs, tf_idf_threshold, gram) = data
    df =  np.count_nonzero(gram) / total_docs

    if df < tf_idf_threshold/len(docs):
        return (unique, [])
    
    return (unique, gram)


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
