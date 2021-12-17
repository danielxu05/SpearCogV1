

import numpy as np
from pyibl import Agent, similarity
import pickle
from .Agent_util import *


class SpearCogAgent:
    def __init__(self, userid, corpus, penalty = 30):
        self.userid = userid
        self.IBLAgent = Agent(userid,['email','decision'])
        self.corpus = corpus
        similarity(self.corpus.getSim, "email")
        self.IBLAgent.mismatch_penalty = penalty
        self.performance= None

    def fit(self, data,training_rep=1):
        #fit the agent with instances 
        #train agent 
        for _ in range(training_rep):
            for (text,label) in data:
                index = text.index
                self.IBLAgent.populate(1,[index,label])
                self.IBLAgent.populate(-1,[index,label.negate()])

    def test(self, data):
        #test on the known cases
        # return accruacy
        groundtruth,predicted = [],[]

        for (text,label) in data:
            choice = self.predict(text)
            groundtruth.append(label._value_)
            predicted.append(choice[1]._value_)
        self.performance = getPerformance(groundtruth,predicted)
        self.performance['size'] = len(groundtruth)
        return self.performance

    def ReportResult(self):
        #report precision recall accuracy 
        return self.performance

    def predict(self,email):
        #predict single instance
        index = email.index
        if index not in self.corpus.texts:
            self.corpus.Add_text(email)
       # print('Predict function add index:',self.corpus,index)
        choice = self.IBLAgent.choose([index, Action.RESPONSE],[index,Action.IGNORE])
        self.IBLAgent.respond(0)
        return choice
    
    def saveAgent(self,path):
        #save agents 
        nfile = open(path+self.userid,'wb')
        pickle.dump(self,nfile)
        
#    @classmethod
def loadAgent(path):
    nfile = open(path,'rb')
    return pickle.load(nfile)

def saveAgent(object,path):
    #save agents 
    nfile = open(path,'wb')
    pickle.dump(object,nfile)
    nfile = open(path+'_corpus','wb')
    pickle.dump(object.agents[0].corpus,nfile)

class GroupSpearCogAgent:
    def __init__(self):
        self.agents = []
    
    def addAgent(self,agent):
        self.agents.append(agent)
        self.weights = self.getAccWeights()

    
    def predict(self,email):
        """
        input: 
        email: email object
        output: return a list of choices made by agents
        """
        result = []
        for agent in self.agents:
            #print('agent corpus address printed from group agent class',agent.corpus)
            choice = agent.predict(email)[1]
            result.append(choice)
        return result
    
    def getGroupResponse(self,email,threshold = 0.5,weighted = False):
        resultarr = []
        result = self.predict(email)
        resultarr = [decision._value_ for decision in result]
        resultarr = np.array(resultarr)
        if weighted:
            groupresult = self.weights.dot(resultarr)
        else:
            groupresult = np.mean(resultarr)
        return 1 if groupresult>=threshold else 0,groupresult

    def getAccWeights(self):
        weights = []
        for agent in self.agents:
            weights.append(agent.performance['Accuracy'])
        weights = np.array(weights)
        exp = np.exp(weights)
        return exp/np.sum(exp)
    
    
