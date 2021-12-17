import pandas as pd
from agent import GroupSpearCogAgent

from agent import SpearCogAgent
from corpus import Corpus, Email
from sentence_transformers import SentenceTransformer
from Agent_util import *
DATAPATH = '../data/'

emails = pd.read_csv(DATAPATH+'combined_userclassification.csv')
emails['Subject'] = emails['Subject'].astype('str')
emails['Email'] = emails['Email'].astype('str')

train_ratio = 0.8


userids = ['22b6d145931f099480ff6cf8288d547c',
        '686288c21b1c357c8a9187c923abf116',
        'b2f8fa38faa90c17d2195662d7c7598f',
        'daaa1d9e709c0b6deba687602d958b8e',
        'c5cb408dbea72628b31173c1e7dd7b6c',
        '50bd97113a3b5ccbbffa3cafd1c23cc1',
        '4f378c04f6e27e7c46f74059532c49bd',
        '94ffb8105c99c7e7046d14bbb5b4cec3',
        '96bff063e57ffc9875d6ccc5fc4b6bec',
        '681ca6381d404457e294fadd6ea0343b']

##
BertModel = SentenceTransformer(DATAPATH+'model1')
corpus = Corpus(BertModel)
groupagent = GroupSpearCogAgent()
for userid in userids[:1]:
    spearagent = SpearCogAgent(userid,corpus)


    auser = emails[emails['RaterID']==userid]

    auser = auser.sample(frac=1)
    trainset,testset = [],[]

    for i in range(int(train_ratio*len(auser))):
        temp = auser.iloc[i]
        email = Email(temp['Subject'],temp['Email'])
        trainset.append((email, encode_decision(temp['Response_'])))
        corpus.Add_text(email)

    for i in range(int(train_ratio*len(auser)),len(auser)):
        temp = auser.iloc[i]
        email = Email(temp['Subject'],temp['Email'])
        testset.append((email,encode_decision(temp['Response_'])))
        corpus.Add_text(email)
    spearagent.fit(trainset)
    spearagent.test(testset)
    print(spearagent.ReportResult())
    groupagent.addAgent(spearagent)
#spearagent.saveAgent()
