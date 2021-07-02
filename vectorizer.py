from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(text):
    vectorizer = TfidfVectorizer(stop_words='english', 
    max_features= 1000, # keep top 1000 terms 
    max_df = 0.5, 
    smooth_idf=True)

    return vectorizer.fit_transform(text)