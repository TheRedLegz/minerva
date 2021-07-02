from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import pandas as pd
import re

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
w_tokenizer = TweetTokenizer()

def clean_tweet(text):
    text = remove_links(text)
    text = remove_hashtags(text)
    text = remove_special_characters(text)
    text = remove_stop_words(text)
    # Switch between stem and lemmatize when necessary
    text = lemmatize_text(text)
    # text = stem_text(text)

    text = detokenize_text_list(text)

    return text

def stem_text(text_list):
    return [(ps.stem(w)) for w in text_list]

def lemmatize_text(text_list):
    return [(lemmatizer.lemmatize(w)) for w in text_list]

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    text_list = []
    for w in w_tokenizer.tokenize(text):
        if w not in stop_words:
            text_list.append(w)
    return text_list

def remove_links(text):
    return re.sub(r"http\S+", "", text)

def remove_hashtags(text):
    return re.sub("[#@]+\S+", "", text)
    # return re.sub('[@#!,.;:?/]', '', text)

def remove_special_characters(text):
    return re.sub('[@#!,.;:?/]', '', text)

def detokenize_text_list(text_list):
    detokenized_text = ' '.join(text_list)
        
    return detokenized_text
    
if __name__ == "__main__":
    # df = pd.read_csv("Tweets.csv")
    # for x in range(df['text'].size):
    #     df['text'][x] = re.sub('[@#!,.;:?/]', '', df['text'][x])
    #     df['text'][x] = stem_text(df['text'][x])

    test_tweet = "I’ll know if there is a weeb on the beach if they recognize my bikini bottoms https://pic.twitter.com/3xWXnT4tKb"
    test_tweet = clean_tweet(test_tweet)

    print(test_tweet)   