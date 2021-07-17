from operator import index
import pandas as pd
import re
# from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

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
    return tokenized_doc

    detokenized_doc = []
    for i in range(len(tweets_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)
    # topicModeling(detokenized_doc)
    # tweets_df['clean_doc'] = detokenized_doc

    return detokenized_doc


# def bgram(document_array):
#     vectorizer = TfidfVectorizer(ngram_range=(2,2), sublinear_tf=True, use_idf=True)
#     sparse_matrix = vectorizer.fit_transform(document_array)
#     features = vectorizer.get_feature_names()

#     sum = sparse_matrix.sum(axis=0)
#     data = []

#     for i, j in enumerate(features):
#         data.append((j, sum[0, i]))

#     return data

# TODO test accuracy of LSA
# TODO try connecting LSA to SOM
# TODO make SOM

def topicModeling(document_array):

    # X = vectorizer.fit_transform(document_array)
    # features = vectorizer.get_feature_names()

    # svd_model = TruncatedSVD(n_components=26, n_iter=100)
    # svd_model.fit(X)
    # print(features)
    
    # processed_corpus = preprocess_documents(document_array)

    """ 

    step 1: preprocess documents (stopwords, lemmatize)
    step 2: split to bigram
    step 3: generate document-term matrix
    step 4: apply lsa to the document-term matrix
    
    """

    # gensim tf idf <= array of bigrammed documents



    # array of tokenized documents


    # pprint(document_array)


    bigram = Phrases(document_array, min_count=1, threshold=2)
    
    bigram_model = Phraser(bigram)

    res = []

    for document in document_array:
        res.append(bigram_model[document])
    dictionary = Dictionary(res)
    
    # pprint(res)

    bow_corpus = [dictionary.doc2bow(text) for text in res]

    # pprint(dictionary.save_as_text("dictionary", sort_by_word=False))
    pprint(bow_corpus)
    # pprint(len(bow_corpus))
 

    tfidf = TfidfModel(bow_corpus, smartirs='npu')
    corpus_tfidf = tfidf[bow_corpus]
    
    coherenceList_UMass = []
    numTopicsList = [20,100,200,300,400,500,800,1000,1500]
    
    for k in numTopicsList:
        c_UMass = compute_coherence_UMass(corpus_tfidf, dictionary, k)
        coherenceList_UMass.append(c_UMass)
    # print(coherenceList_UMass)
    
    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(svd_model.components_)
    # # principalDf = pd.DataFrame(data=principalComponents)
    # return principalComponents


    # return principalDf
    
def compute_coherence_UMass(corpus, dictionary, k):
    lsi_model = LsiModel(corpus=corpus, num_topics=k)
    # print(lsi_model.print_topic(topicno=1))
    coherence = CoherenceModel(model=lsi_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    return coherence.get_coherence()





# def topicModeling():
#     vectorizer = TfidfVectorizer(stop_words='english', 
#     max_features= 1000, # keep top 1000 terms 
#     max_df = 0.5, 
#     smooth_idf=True)
#     X = vectorizer.fit_transform(tweets_df['clean_doc'])
#     # print(X.shape)

#     svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
#     svd_model.fit(X)
#     terms = vectorizer.get_feature_names()

#     # print(len(terms))
#     # for i, comp in enumerate(svd_model.components_):
#     #     terms_comp = zip(terms, comp)
#     #     sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
#         # print("Topic "+str(i)+": ")
#         # for t in sorted_terms:
#         #     print(t)
#         #     print(" ")
#         # print(sorted_terms)

#     pca = PCA(n_components = 2)
#     principalComponents = pca.fit_transform(svd_model.components_)
#     principalDf = pd.DataFrame(data = principalComponents, columns = ['1', '2'])
    
#     return principalDf

texts = preprocessing()
res = topicModeling(texts)