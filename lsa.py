import numpy
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser
from gensim.corpora.dictionary import Dictionary
from gensim.models import LsiModel, TfidfModel
from pprint import pprint
from IPython.display import display

tweets_df = pd.read_json (r'sample.json')


def preprocessing():
    tweets_df['clean_doc'] = tweets_df['full_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    tweets_df['clean_doc'] = tweets_df['clean_doc'].str.replace("[^a-zA-Z#]", " ", regex=True)
    tweets_df['clean_doc'] = tweets_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    tweets_df['clean_doc'] = tweets_df['clean_doc'].apply(lambda x: x.lower())
    tokenized_doc = tweets_df['clean_doc'].apply(lambda x: x.split())
    detokenized_doc = []
    
    for i in range(len(tweets_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)
        
    tweets_df['clean_doc'] = detokenized_doc
    return detokenized_doc


def lsiGensim(detokenized):
    processed_corpus = preprocess_documents(detokenized)

    bigram = Phrases(processed_corpus, min_count=1, threshold=2)
    bigram_model = Phraser(bigram)
    
    res = []
    for doc in processed_corpus:
        res.append(bigram_model[doc])

    dictionary = Dictionary(res)

    bow_corpus = [dictionary.doc2bow(text) for text in res]
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
    s = numpy.array(optimized.projection.s)
    x = numpy.arange(len(s), step=1, dtype=numpy.int8)
    plt.bar(x, s)
    # singular values
    plt.show()
    df = pd.DataFrame(list(optimized[corpus_tfidf]))
    # document topic matrix
    display(df)
    df = pd.DataFrame(optimized.projection.u[:,:5])
    # word topic matrix
    display(df)
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
    return res