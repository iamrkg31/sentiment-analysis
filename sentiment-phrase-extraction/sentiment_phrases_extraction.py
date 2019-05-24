import re
import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer

stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_lg')


# (JJS|JJR|JJ) (NNPS|NNP|NNS|NN) (NNPS|NNP|NNS|NN)
# (JJS|JJR|JJ) (NNPS|NNP|NNS|NN)
# (RBR|RBS|RB) (JJS|JJR|JJ)
# (RBR|RBS|RB) (JJS|JJR|JJ) (NNPS|NNP|NNS|NN)
# (RBR|RBS|RB) (VBD|VBG|VBN|VBP|VBZ|VB)
# (RBR|RBS|RB) (RBR|RBS|RB) (JJS|JJR|JJ)
# (VBD|VBG|VBN|VBP|VBZ|VB) (NNPS|NNP|NNS|NN)
# (VBD|VBG|VBN|VBP|VBZ|VB) (RBR|RBS|RB)


sentiment_pos_tag_patterns = "([a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ)|[a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(VBD|VBG|VBN|VBP|VBZ|VB)|[a-zA-Z0-9'-]*_(VBD|VBG|VBN|VBP|VBZ|VB) [a-zA-Z0-9'-]*_(RBR|RBS|RB)|[a-zA-Z0-9'-]*_(VBD|VBG|VBN|VBP|VBZ|VB) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN))"
pos_lexicons = pd.read_csv("../data/positive-words.txt")["word"].tolist()
neg_lexicons = pd.read_csv("../data/negative-words.txt")["word"].tolist()
negations = "( |^)(n't|not|never|neither|none|no)( |$)"


def pos_tags(sentence):
    """Gets POS tags of given text"""
    sentence = re.sub("’", "'", sentence)
    tokens = nlp(sentence)
    tags = []
    for tok in tokens:
        tag = re.sub("\$","@",tok.tag_)
        tags.append(tok.text+"_"+tag)
    return " ".join(tags)


def extract_tagged_phrases(text):
    """Extracts tagged phrases using patterns"""
    found_iter = re.finditer(sentiment_pos_tag_patterns, text)
    found = list(set([i.group() for i in found_iter]))
    return found


def check_pos_neg_lexicons(phrase):
    """Checks tagged phrases for positive and negative words"""
    words = phrase.lower().split()
    flag = False
    for w in words:
        if w in pos_lexicons or w in neg_lexicons:
            flag = True
            break
    return flag


# import data
df = pd.read_csv("../data/test.csv")

# tag the sentences of the msgs
df["Sentences"] = df["tweet"].apply(lambda x: PunktSentenceTokenizer().tokenize(str(x)))
s = df.apply(lambda x: pd.Series(x["Sentences"]),axis=1).stack().reset_index(level=1, drop=True)
s.name = "Sentence"
df = df.drop('Sentences', axis=1).join(s)
df = df.reset_index()
df["Tagged Sentence"] = df["Sentence"].apply(lambda x: pos_tags(str(x).lower()))

# get tagged phrases
df["Tagged Phrase List"] = df["Tagged Sentence"].apply(lambda x: extract_tagged_phrases(str(x)))
df = df[df["Tagged Phrase List"].str.len() != 0]
s = df.apply(lambda x: pd.Series(x["Tagged Phrase List"]),axis=1).stack().reset_index(level=1, drop=True)
s.name = "Tagged Phrase"
df = df.drop('Tagged Phrase List', axis=1).join(s)

# Get phrases
df["Phrase"] = df["Tagged Phrase"].apply(lambda x: re.sub("_[A-Z@]*","",x))

# Keep sentiment phrases only
df["Flag"] = df["Phrase"].apply(lambda x: check_pos_neg_lexicons(x))
df = df[df["Flag"]]

# Write to file
df.to_csv("../data/sentiment_phrases.csv", index=False)


############## For categorizing sentiment pharses to positive, negative #####################

# def categorize_sentiment_phrases(phrase):
#     if re.search(negations, phrase):
#         return False
#     words = phrase.lower().split()
#     flag = False
#     for w in words:
#         if w in pos_lexicons:
#             flag = True
#             break
#     return flag
#
# df = pd.read_csv("sentiment_phrases.csv")
# df["Category"] = df["Phrase"].apply(lambda x: categorize_sentiment_phrases(x))
# pos_phrases = df[df["Category"]]["Phrase"].tolist()
# neg_phrases = df[~df["Category"]]["Phrase"].tolist()
