import random
from tqdm import tqdm
train_rep = 1000
penalties = [1,5,30]#,50,90] 
#penalties = [90]

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from pyibl import Agent, similarity
from random import random
import numpy as np

accruacy=[]
labels = ['Accuracy','Percision','Recall']
similar = {}


def negate(r):
    if r =='Response':
        return 'Ignore'
    else:
        return 'Response'

def getPerformance(a,b):
    return [accuracy_score(a,b),precision_score(a,b,pos_label ='Response'),recall_score(a,b,pos_label='Response')]

## IBL Agent
def train_agent(userclassification,userid,seeds,Simfunction,randonmization = True,survey = False, train_ratio=0.8,train_rep = 1):
    auser = userclassification[userclassification['RaterID']==userid]#.sort_values('Starttime')
    subaccruacy = {'UserID':userid}
    for penalty in penalties:
        performance = []
        for seed in seeds:
            groundtruth,predicted = [],[]
            #randomize the order
            if randonmization:
                np.random.seed(seed)
                auser = auser.sample(frac = 1)
                #np.random.seed(None)
            agent = Agent(userid, ['email','decison'])
            similarity(Simfunction,"email")
            agent.mismatch_penalty = penalty
            #survey need to comment in 
            if survey:
                #use the survey response or unique emails; Reset index would l
                auser['index'] =auser.index
            array = auser[['index','Response_']].values            
            train,valid = array[:int(train_ratio*len(array))],array[int(len(array)*train_ratio):]
            
            for _ in range(train_rep):
#                random.shuffle(train)
                for x in train:
                    #print(x)
                    agent.populate(1, list(x))
                    agent.populate(-1, [x[0],negate(x[1])])                    
            for x in valid:
                choise = agent.choose(list(x),[x[0],negate(x[1])])
                groundtruth.append(x[1])
                predicted.append(choise[1])
                #print(choise,x)
                agent.respond(0)
            performance.append(getPerformance(groundtruth,predicted))
        performance = np.mean(np.array(performance),axis =0)
        performance_var = np.var(np.array(performance),axis =0)
        for i in range(len(labels)):
            subaccruacy[str(penalty)+'_'+labels[i]] = performance[i]
        for i in range(len(labels)): 
            subaccruacy[str(penalty)+'_var_'+labels[i]] = performance[i]/len(seeds)
    #array.append(subaccruacy)
    return subaccruacy

