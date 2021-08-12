import re
import emoji
import nltk
import requests
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content

stop_words = set(stopwords_list.decode().splitlines()) 
lemmatizer = WordNetLemmatizer()

dd = []

def preprocess(string):
    def remove_links(string):
        res = re.sub(r'http\S+', '', string)
        return res
    
    def remove_emojis(string):
        return emoji.get_emoji_regexp().sub(u'', string)
    
    def remove_numbers(string):
        res = re.sub(r'\d+', ' ', string)
        return res

    def remove_non_ascii(string):
        return string.encode("ascii", "ignore").decode()

    def remove_punctation(string):
        res = re.sub(r'[^\w\s]', ' ', string)
        res = re.sub('’s', ' ', res)
        res = re.sub('’ve', ' have ', res)
        return res

    def remove_html_tags(string):
        res = re.sub(r'&(gt|lt|amp|nbsp|quot|apos|cent|pound|yen|euro|copy|reg);', '', string)
        return res

    # NOTE this returns an array
    def remove_stop_words(string):
        string = re.sub(r"\n", " ", string)
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

    def remove_mentions(string):
        res = re.sub(r'@[a-zA-Z0-9_]+', ' ', string)
        return res

    def remove_hashtags(string):
        res = re.sub(r'#[a-zA-Z0-9]+', ' ', string)
        return res

    try:
        string = string.lower()
        string = remove_links(string)
        string = remove_mentions(string)
        string = remove_hashtags(string)
        string = remove_emojis(string)
        string = remove_numbers(string)
        string = remove_html_tags(string)
        string = remove_non_ascii(string)
        string = remove_punctation(string)
        string = remove_stop_words(string)
        string = lemmatize_string(string)
    except:
        print('error in string', string)
        return

    return string.strip()



def preprocess_documents(array):
    res = []

    for a in array:
        res.append(preprocess(a))

    return res



def write_to_file(fn, array, is_2d=False):
    with open(fn, 'w', encoding="utf8") as file:
        file.write("Number of entries: {}\n\n".format(len(array)))

        if not is_2d:
            for i, a in enumerate(array):
                string = "$Index {}\n{}\n\n".format(i, a)
                file.write(string)
        else:
            for i, a in enumerate(array):
                file.write('Row {}: {} cols\n'.format(i, len(a)))

                for val in a:
                    file.write('{}\t'.format(val))

                file.write('\n\n')