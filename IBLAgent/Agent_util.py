
from mmap import ACCESS_COPY
from sentence_transformers import util
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from enum import Enum

import re
import nltk

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
add_stop_words = ['nbsp','n','p','r',';','&']
stop_words.extend(add_stop_words)

def preprocess_text(sentence,punctuation = True):
    # Removing html tags
    sentence = remove_tags(sentence)
#    sentence = process_NER(sentence)
    # Remove punctuations and numbers
    if punctuation:
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub('nbsp',' ',sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    tokens = wpt.tokenize(sentence)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    sentence = ' '.join(filtered_tokens)
    return sentence
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)
def trim_string(x,first_n_words):
    x = x.split(maxsplit=first_n_words)
    return ' '.join(x[:first_n_words]).replace('\n', '').replace('\r', '')

class Action(Enum):
    RESPONSE = 1
    IGNORE = 0

    def negate(self):
        if self == Action.RESPONSE:
            return Action.IGNORE
        elif self == Action.IGNORE:
            return Action.RESPONSE

def BERT_Sim(sentence1,sentence2,model):
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_scores.item()

def getPerformance(y_true,y_pred):
    return {'Accuracy':accuracy_score(y_true,y_pred),
            'Percision':precision_score(y_true,y_pred),
            'Recall':recall_score(y_true,y_pred),
            'f1':f1_score(y_true,y_pred)}


def encode_decision(decision):
    if decision =='Response':
        return Action.RESPONSE
    elif decision =='Ignore':
        return Action.IGNORE
    else:
        raise ValueError

        