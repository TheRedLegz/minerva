import nltk
import matplotlib.pyplot as plt
import numpy as np
import string as str
import pandas as pd
import math
import re
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
# from pprint import pprint
# from IPython.display import display

# nltk.download('stopwords')
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


dd = []
def test_print():
  print("TEST")

def preprocess(string):
    def remove_links(string):
        res = re.sub(r'http\S+', '', string)
        return res
    
    def remove_emojis(string):
        return emoji.get_emoji_regexp().sub(u'', string)
    
    def remove_numbers(string):
        res = re.sub(r'\d+', '', string)
        return res

    def remove_non_ascii(string):
        return string.encode("ascii", "ignore").decode()

    def remove_punctation(string):
        res = string.translate(string.maketrans("", "", str.punctuation))
        res = re.sub('’s', '', res)
        res = re.sub('’ve', ' have', res)
        return res

    def remove_html_tags(string):
        res = re.sub(r'&(gt|lt|amp|nbsp|quot|apos|cent|pound|yen|euro|copy|reg);', '', string)
        return res

    # NOTE this returns an array
    def remove_stop_words(string):
        split = string.split(' ')
        res = [w for w in split if w not in stop_words]
        return ' '.join(res)

    def pos_tag(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()

        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        
        return tag_dict.get(tag, wordnet.NOUN)

    def lemmatize_string(string):
        split = string

        if not isinstance(string, list):    
            split = string.split(' ')

        res = []

        for word in split:
            possible_lemmas = wordnet._morphy(word, pos=pos_tag(word))

            if possible_lemmas:
                to_add = None

                for a in possible_lemmas:
                    if a in dd:
                        to_add = a

                
                if to_add is None:
                    to_add = min(possible_lemmas, key=len)
                    dd.append(to_add)

                res.append(to_add)

            else:
                if word not in dd:
                    dd.append(word)

                res.append(word)

        return ' '.join(res)

    string = string.lower()
    string = remove_links(string)
    string = remove_emojis(string)
    string = remove_numbers(string)
    string = remove_html_tags(string)
    string = remove_non_ascii(string)
    string = remove_punctation(string)
    string = remove_stop_words(string)
    # string = lemmatize_string(string)

    return string.strip()



def bag_of_words(document_array):

    doc_grams = []

    for doc in document_array:
        string = preprocess(doc)
        
        if len(string) == 0:
            continue

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
            idf = math.log((doc_count / (has_word_count)))

            matrix[i][j] = tf * idf 
    return matrix

def lsaSklearn(tfidf_matrix):
    (x, y) = tfidf_matrix.shape
    lsa = TruncatedSVD(n_components= y-1, algorithm="randomized", n_iter=5, random_state= 42)
    lsa.fit(tfidf_matrix)

    cumsum = lsa.explained_variance_ratio_.cumsum()
    optimal_num = y - 1

    for i, a in enumerate(cumsum):
        if a > .8:
            optimal_num = i + 1
            break
    

    lsa_final = TruncatedSVD(n_components=optimal_num, algorithm="randomized", n_iter=5, random_state= 42)
    lsa_final.fit(tfidf_matrix)

    sigma = np.diag(lsa_final.singular_values_)
    v_t = lsa_final.components_
    
    res = np.dot(sigma, v_t)
    return np.transpose(res)
    
def pca(matrix):
    (x, y) = matrix.shape
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    
    pca = PCA(n_components=y)
    pca.fit(scaled)
    cumsum = pca.explained_variance_ratio_.cumsum()

    optimal_components = y
    
    for i, sum in enumerate(cumsum):
        if sum > .80:
            optimal_components = i + 1
            break

    pca_final = PCA(n_components=optimal_components)
    res = pca_final.fit_transform(scaled)

    return res

def most_common(lst, n):
  if len(lst) == 0:
      return -1
      
  counts = np.zeros(shape=n, dtype=np.int)

  for i in range(len(lst)):
    counts[lst[i]] += 1

  return np.argmax(counts)

def find_bmu(matrix, input, matrix_size):
    (row, col) = matrix_size

    num_features = len(input)

    bmu = (0,0)
    nearest = 10000000

    for i in range(row):
        for j in range(col):

            distance = euc_distance(input, matrix[i][j], num_features)

            if distance < nearest:
                nearest = distance
                bmu = (i, j)

    return bmu

def euc_distance(input, cell, num_features):
    sum = 0

    for n in range(num_features):
        sum = sum + pow(input[n] - cell[n], 2)
    
    return math.sqrt(sum)

def man_distance(bmu_row, bmu_col, row, col):
    return np.abs(bmu_row-row) + np.abs(bmu_col-col)

def SOM(data, learn_rate, matrix_size):

    # 1 Initialize some constants

    (steps, num_features) = data.shape
    (row, col) = matrix_size
    range_max = row + col 

    # 2 Create the initial matrix

    matrix = np.random.random_sample(size=(row,col,num_features))

    # 3 Run main logic

    steps_max = 5000 

    indices = np.zeros(len(data))

    for s in range(steps_max):
        if s % (steps_max/50) == 0: print(str((s/steps_max) * 100.00) + " percent")
        
        # Update
        percent = 1.0 - ((s * 1.0) / steps_max)

        # curr_range for manhattan distance updating
        curr_range = (int)(percent * range_max)
        
        # curr_range for pythagorean updating
        # curr_range = range_max * exp(-(s / steps_max))
        curr_rate = percent * learn_rate

        # get a unique input from data

        rand = np.random.randint(len(data))

        input = data[rand]

        # get the bmu
        (bmu_row, bmu_col) = find_bmu(matrix, input, matrix_size)

        for i in range(row):
            for j in range(col):

                cell = matrix[i][j]
                
                # Weight Updating (w/ man_distance)
                if man_distance(bmu_row, bmu_col, i, j) < curr_range:
                    matrix[i][j] = cell + curr_rate * (input-cell)

                    # [1, 0.4, 0.5, ..., 9n]
                    # [0.3, 0.6, 1, ..., 9n]

                # Weight Updating (w/o man_distance)
                # Formula: cell+curr_rate*(EXP(-((POWER(bmui-i,2)+POWER(bmuj-j,2))/(num_features???*POWER(curr_range,2)))))*(input- cell)
                # matrix[i][j] = cell  + curr_rate * (exp(-(((bmu_row - i)**2 + (bmu_col - j)**2)/(num_features*(curr_range**2))))) * (input - cell)

    return matrix

def construct_u_matrix(SOM_matrix, matrix_size):

    (row, col) = matrix_size
    u_matrix = np.zeros(shape=matrix_size, dtype=np.float64)

    for i in range(row):
        for j in range(col):
            
            v = SOM_matrix[i][j]

            sum_dists = 0.0
            ct = 0
            
            if i-1 >= 0:  
                sum_dists += euc_distance(v, SOM_matrix[i-1][j], len(v)); ct += 1

            if i+1 <= row-1:
                sum_dists += euc_distance(v, SOM_matrix[i+1][j], len(v)); ct += 1

            if j-1 >= 0:  
                sum_dists += euc_distance(v, SOM_matrix[i][j-1], len(v)); ct += 1

            if j+1 <= col-1: 
                sum_dists += euc_distance(v, SOM_matrix[i][j+1], len(v)); ct += 1

            
            u_matrix[i][j] = sum_dists / ct

    plt.imshow(u_matrix, cmap='gray') 
    plt.show()