import math
import numpy as np
from pprint import pprint

def bag_of_words(document_array):

    # NOTE maybe convert the document array into a list of tokenized documents
    
    unique_words = []

    for doc in document_array:
        words = doc.split(' ')

        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    unique_count = len(unique_words)
    document_count = len(document_array)

   
    # bow = words as row, document as col
    bow = np.zeros((unique_count, document_count), dtype=int)

    for i in range(unique_count):
        word = unique_words[i]

        for j in range(document_count):
            doc = document_array[j]
            doc_words = doc.split(' ')

            count = doc_words.count(word)

            bow[i][j] = int(count)
        

    return (bow, unique_words)

def tf_idf(document_array, bow = None):
    # NOTE bigram - use package

    if bow is None:
        (bow, _) = bag_of_words(document_array)

    """ 
            1   2   3   4   doc_n
    CAT     1   0   0   1   0
    word    0   0   0   1   0
    word    0   0   0   1   0
    word    0   0   0   1   0
    word    0   0   0   1   0
    word    0   0   0   1   0
    
    """


    """ 
            1   2   3   4   doc_n
    CAT     1   0   0   0   1
    word    0   0   0   0   0
    word    0   0   0   0   0
    word    0   0   0   0   0
    word    0   0   0   0   0
    word    0   0   0   0   0
    
    """

    """ 
    TF = word occurence in the document / total number of words in the document
    IDF = total number of documents / number of documents where the word occurs
    
    """

    matrix = np.zeros(bow.shape, dtype=float)
    doc_count = len(document_array)


    for i, row in enumerate(bow):
        idf = math.log(doc_count / int(doc_count - list(row).count(0) + 1))

        for j, col in enumerate(row):
            doc = document_array[j]

            # this could still be optimized. maybe use col sum
            word_count = len(doc.split(' '))

            tf = col / word_count
            
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


(bow, dictionary) = bag_of_words(data)

matrix = tf_idf(data)

print(bow)
print(dictionary)
print(matrix)