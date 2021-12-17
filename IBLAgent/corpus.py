from liwc import Liwc
import pickle
from .Agent_util import *
import re
import numpy as np

liwc = Liwc('./data/LIWC2015 Dictionary.dic')
nfile = open('./data/field_names_V2_model2.pkl', 'rb')
liwc_category = pickle.load(nfile)

class Text:
    def __init__(self,text) -> None:
        self.text = preprocess_text(text)
        self.index = hash(self.text)
    
    def combine(self):
        return self.text

class Email:
    def __init__(self,subject,body):
        self.subject = subject
        self.body = body
        self.preprocess()
        self.index = hash(self.subject + ' [SEP] ' + self.body)
    
    def combine(self):
        return self.subject + ' [SEP] ' + self.body
    
    def preprocess(self):
        self.subject = preprocess_text(self.subject)
        self.body = preprocess_text(self.body)

class EmailV2(Email):
    def __init__(self, subject, body, content):
        """
        Subject and Body are from Email
        Content is the agent's information and stored in a dictionary {'Name':'','Company':'','Bank':''}
        """
        super().__init__(subject, body)
        self.LIWC = self.getLIWC()
        self.containName = self.searchWord(content['Name'])
        self.containCompany = self.searchWord(content['Company'])
        self.containWrongBank = self.searchWord(content['Bank']) 
        self.features = []
        self.getLIWC()
        self.features.extend(self.liwc_features)
        self.features.extend([self.containCompany,self.containName,self.containWrongBank])
        self.features = np.array(self.features).astype(np.float32)
        
    def searchWord(self,words):
        wordlist = words.split(' ')
        for word in wordlist:
            if re.search(word,self.subject+ ' '+self.body):
                return True
        return False

    def getLIWC(self):
        text = self.subject + ' ' + self.body
        self.length = len(text.split(' '))
        liwc_features = liwc.parse(text.split(' '))
        self.liwc_features = [0 if liwc_feature not in liwc_features else liwc_features[liwc_feature]/self.length for liwc_feature in liwc_category]
    
    def combine(self):
        return [(self.subject + ' [SEP] ' + self.body, self.features)]

class Corpus:
    def __init__(self,bertmodel):
        self.texts = {}
        self.similarity = {}
        self.simmodel = bertmodel
    
    def Add_text(self, email):
        #add text into corpus if text not appear before
        if email.index not in self.texts:
            self.texts[email.index] = email

    def getSim(self,idx1,idx2):
        #
#        print('get sim corpus id:',self)       
        key = str(idx1) + '_' + str(idx2)
        if key not in self.similarity:
            text1, text2 = self.texts[idx1].combine(), self.texts[idx2].combine() 
            sim = BERT_Sim(text1,text2,self.simmodel)
            self.similarity[key] = (sim+1)/2
        return self.similarity[key]
