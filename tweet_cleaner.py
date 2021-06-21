from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import pandas as pd
import re

ps = PorterStemmer()
w_tokenizer = TweetTokenizer()

def stem_text(text):
    return [(ps.stem(w)) for w in \
                                     w_tokenizer.tokenize((text))]

if __name__ == "__main__":
    df = pd.read_csv("Tweets.csv")
    for x in range(df['text'].size):
        df['text'][x] = re.sub('[@#!,.;:?/]', '', df['text'][x])
        df['text'][x] = stem_text(df['text'][x])

    print(df['text'][14629])    