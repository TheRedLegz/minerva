from operator import index
import numpy
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from gensim.parsing.preprocessing import preprocess_documents
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser
from gensim.corpora.dictionary import Dictionary
from gensim.models import LsiModel, TfidfModel


from pprint import pprint
import math
# nltk.download('stopwords')

tweets_df = pd.read_json (r'sample.json')


def preprocessing():
    # stop_words = stopwords.words('english')
    tweets_df['clean_doc'] = tweets_df['full_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    tweets_df['clean_doc'] = tweets_df['clean_doc'].str.replace("[^a-zA-Z#]", " ", regex=True)
    tweets_df['clean_doc'] = tweets_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    tweets_df['clean_doc'] = tweets_df['clean_doc'].apply(lambda x: x.lower())
    tokenized_doc = tweets_df['clean_doc'].apply(lambda x: x.split())
    # tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    detokenized_doc = []
    for i in range(len(tweets_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)

    tweets_df['clean_doc'] = detokenized_doc
    return detokenized_doc

# def topicModeling(detokenized):
#     vectorizer = TfidfVectorizer(ngram_range=(1,2),stop_words='english', 
#     max_features= 1000, # keep top 1000 terms 
#     max_df = 0.5, 
#     smooth_idf=True)
#     X = vectorizer.fit_transform(detokenized)
#     # print(X.shape)

#     svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
#     svd_model.fit_transform(X)
#     # terms = vectorizer.get_feature_names()

#     # print(len(terms))
#     # for i, comp in enumerate(svd_model.components_):
#     #     terms_comp = zip(terms, comp)
#     #     sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=False)[:7]
#     #     print("Topic "+str(i)+": ")
#     #     for t in sorted_terms:
#     #         print(t)
#     #         print(" ")
#     #     print(sorted_terms)
#     pprint(svd_model.components_)
#     print(len(svd_model.components_))
#     transpose = numpy.transpose(svd_model.components_)
#     # pprint(transpose)
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(transpose)

#     # pca = PCA(n_components = 2)
#     # principalComponents = pca.fit_transform(svd_model.components_)
#     # principalDf = pd.DataFrame(data = principalComponents, columns = ['1', '2'])
    
#     return transpose

def lsiGensim(detokenized):
    processed_corpus = preprocess_documents(detokenized)

    # do bigram
    # input bigram into the dictionary

    dictionary = Dictionary(processed_corpus)

    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    tfidf = TfidfModel(bow_corpus, smartirs='npu')

    corpus_tfidf = tfidf[bow_corpus]


    coherenceList_UMass = []
    numTopicsList = [35,36,37,38,39,40]
    for k in numTopicsList:
        c_UMass = compute_coherence_UMass(corpus_tfidf, dictionary, k)
        coherenceList_UMass.append(c_UMass)
    plt.plot(numTopicsList, coherenceList_UMass)
    minpos = coherenceList_UMass.index(min(coherenceList_UMass))


    optimized = LsiModel(corpus=corpus_tfidf, num_topics=numTopicsList[minpos])

    # research y 22 topics are showing, instead of 30+
    # try to examine the tfidf matrix first
    # 

    topics = optimized.get_topics()
    return topics
    

def compute_coherence_UMass(corpus, dictionary, k):
    lsi_model = LsiModel(corpus=corpus, num_topics=k)
    coherence = CoherenceModel(model=lsi_model,corpus=corpus, dictionary=dictionary,coherence='u_mass')

    return coherence.get_coherence()




def pca(matrix, k):
    # TODO must reduce the number of topics

    # the input must be words x topics
    # k must be the desired num of topics

    # ? maybe scale the things first before doing the PCA
    

    print(matrix.shape)
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(matrix)

    

    pca = PCA(k)

    res = pca.fit_transform(scaled)


    print(pca.explained_variance_ratio_.cumsum())
    print(numpy.array(res).shape)
    return res




texts = preprocessing()


something = [
    "cat cat cat cat",
    "cat some cat some",
    "cute cat haha"
]


matrix = numpy.array(lsiGensim(texts)).T

res = pca(matrix, 16)