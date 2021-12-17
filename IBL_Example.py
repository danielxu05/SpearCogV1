import IBLAgent
import pandas as pd

from sentence_transformers import SentenceTransformer

DATAPATH = 'data/'

#import data 
emails = pd.read_csv(DATAPATH+'combined_userclassification.csv')
emails['Subject'] = emails['Subject'].astype('str')
emails['Email'] = emails['Email'].astype('str')
#data format
# UserID   text   Reaction
#  a        ....    Response
#  a        ....     Ignore
#  b        ....     Response

train_ratio = 0.8
userid = '22b6d145931f099480ff6cf8288d547c'
BertModel = SentenceTransformer('nli-distilroberta-base-v2')

#create a corpus class and bertmodel is a NLP model for text similarity
corpus = IBLAgent.Corpus(BertModel)
agent = IBLAgent.SpearCogAgent(userid= userid,corpus= corpus)
auser = emails[emails['RaterID']==userid]


auser = auser.sample(frac=1)
trainset,testset = [],[]
##prepare train set and test set
for i in range(int(train_ratio*len(auser))):
    temp = auser.iloc[i]
    #consider both subject and email
    email = IBLAgent.Email(subject = temp['Subject'],body =temp['Email'])
    #only consider one phrase of text(email body only in this case)
    #email = IBLAgent.Text(text =temp['Email'])
    trainset.append((email, IBLAgent.encode_decision(temp['Response_'])))
    corpus.Add_text(email)

for i in range(int(train_ratio*len(auser)),len(auser)):
    temp = auser.iloc[i]
    email = IBLAgent.Email(subject = temp['Subject'], body = temp['Email'])
    testset.append((email,IBLAgent.encode_decision(temp['Response_'])))
    corpus.Add_text(email)

agent.fit(trainset)
agent.test(testset)
print(agent.ReportResult())

