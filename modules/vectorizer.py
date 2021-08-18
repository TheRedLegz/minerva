import math
import nltk
import numpy as np
from modules.preprocessor import preprocess

def bag_of_words(document_array, to_preprocess=True):

    doc_grams = []

    for doc in document_array:
        string = ''

        if to_preprocess:
            string = preprocess(doc)
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
    
def prune_bow(bow, tf_idf_treshold = 1):
    (bow_grams, unique, docs) = bow


    original = np.copy(bow_grams)

    # Get Document Frequency == docs with gram / doc #
    for i in range(len(unique)-1, -1, -1):
        df =  np.count_nonzero(original[:, i:i+1]) / len(docs)

        if df < tf_idf_treshold/len(docs):
            unique.pop(i)
            bow_grams = np.delete(bow_grams, i, 1)
            
    for i in range(len(docs)):
        docs[i] = [word for word in docs[i] if word in unique]

    return (bow_grams, unique, docs)

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
