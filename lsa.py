import nltk
import matplotlib.pyplot as plt
import numpy as np
import string as str
import pandas as pd
import math
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from pprint import pprint
from IPython.display import display
import emoji

# nltk.download('stopwords')
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


dd = []


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
        return res

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
    string = lemmatize_string(string)

    # print(string)
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


# DRIVER HERE

raw = pd.read_json('sample.json')
data = []

for i, row in raw.iterrows():
    data.append(row['full_text'])
    

tf_idf_data = tf_idf(data)
lsares = lsaSklearn(tf_idf_data)
pcares = pca(lsares)

pprint(pcares)
pprint(pcares.shape)
