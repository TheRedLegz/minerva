import nltk
import matplotlib.pyplot as plt
import numpy as np
import string as str
import pandas as pd
import math
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import LsiModel, TfidfModel
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

    def remove_punctation(string):
        res = string.translate(string.maketrans("", "", str.punctuation))
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


    string = string.lower().strip()
    string = remove_links(string)
    string = remove_emojis(string)
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

def lsiGensim(doc_gram):
    dictionary = Dictionary(doc_gram)
    bow_corpus = [dictionary.doc2bow(text) for text in doc_gram]

    tfidf = TfidfModel(bow_corpus, smartirs='npu')

    corpus_tfidf = tfidf[bow_corpus]


    coherenceList_UMass = []
    numTopicsList = [35,36,37,38,39,40]
    for k in numTopicsList:
        c_UMass = compute_coherence_UMass(corpus_tfidf, dictionary, k)
        coherenceList_UMass.append(c_UMass)
    plt.plot(numTopicsList, coherenceList_UMass)
    plt.show()
    minpos = coherenceList_UMass.index(min(coherenceList_UMass))


    optimized = LsiModel(corpus=corpus_tfidf, num_topics=numTopicsList[minpos])
    s = np.array(optimized.projection.s)
    x = np.arange(len(s), step=1, dtype=np.int8)
    plt.bar(x, s)
    # singular values
    plt.show()
    df = pd.DataFrame(list(optimized[corpus_tfidf]))
    # document topic matrix
    # display(df)
    df = pd.DataFrame(optimized.projection.u[:,:5])
    # word topic matrix
    # display(df)
    topics = optimized.get_topics()
    return topics
    

def compute_coherence_UMass(corpus, dictionary, k):
    lsi_model = LsiModel(corpus=corpus, num_topics=k)
    coherence = CoherenceModel(model=lsi_model,corpus=corpus, dictionary=dictionary,coherence='u_mass')

    return coherence.get_coherence()


def pca(matrix, k):
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(matrix)

    pca = PCA(k)

    res = pca.fit_transform(scaled)


    x = np.arange(k, dtype=int)
    height = pca.explained_variance_ratio_
    plt.bar(x, height)
    plt.show()
    return res


# DRIVER HERE



raw = pd.read_json('sample.json')
data = []

for i, row in raw.iterrows():
    data.append(row['full_text'])


data = [
    'I want to eat ice cream',
    'Eating ice cream is all I want',
    'My sister ate all my ice cream',
    'Antartica is full of ice'
]   

(bow,x,y)= bag_of_words(data)


lsires = lsiGensim(y)


# pca(lsires, 16)
# pprint(lsires)
# pprint(lsires.shape)