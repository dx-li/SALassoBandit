#Adapted code from https://towardsdatascience.com/multi-armed-bandits-upper-confidence-bound-algorithms-with-python-code-a977728f0e2d
#for initial structure of classes and visualizations

import logging
from abc import (
    ABC,
    abstractmethod,
)

from collections import defaultdict
from types import LambdaType
from typing import List
from uuid import uuid4

#from pyglmnet import GLM

import scipy.sparse as sps

from sklearn import linear_model
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)

class Bandit(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def pull(self):
        ...



class SGLB(Bandit):
    def __init__(self, beta: np.array, link = lambda x: x, sigma: float = 1):
        """
        Simulates stochastic generalized linear bandit.
        Args:
            beta: True parameter
            sparsity: sparisty parameter
            link: link function (default is linear)
            sigma: standard deviation of error
        """

        self.beta = beta
        self.sigma = sigma
        self.id = uuid4()
        self.link = link

    def pull(self, features: np.array) -> tuple:
        """
        Simulate pulling the arm of the bandit. Error is generated N(0, self.sigma)
        Args:
            features: feature vector
        Output:
            reward, error tuple
        """

        reward = np.dot(features, self.beta)
        reward = self.link(reward)
        error = np.random.normal(loc = 0, scale = self.sigma)

        return reward + error


class NoBanditsError(Exception):
    ...


class BanditHistoryLog:
    def __init__(self):
        self.total_actions = 0
        self.total_rewards = 0
        self.all_rewards = None
        self.all_contexts = []
        self.record = defaultdict(lambda: dict(actions=0, reward=0))

    def record_action(self, bandit, context, reward):
        self.total_actions += 1
        self.total_rewards += reward
        # Ensure that all_rewards end up as a 1D array
        if self.all_rewards is None:
            self.all_rewards = reward
        else:
            self.all_rewards = np.concatenate((self.all_rewards, reward))
        self.all_contexts.append(context)
        self.record[bandit.id]['actions'] += 1
        self.record[bandit.id]['reward'] += reward

    def __getitem__(self, bandit):
        return self.record[bandit.id]


class Agent(ABC):
    def __init__(self):
        self.history_log = BanditHistoryLog()
        self._bandits = None

    @property
    def bandits(self) -> List[Bandit]:
        if not self._bandits:
            raise NoBanditsError()
        return self._bandits

    @bandits.setter
    def bandits(self, val: List[Bandit]):
        self._bandits = val

    @abstractmethod
    def take_action(self):
        ...

    def take_actions(self, n: int):
        for _ in range(n):
            self.take_action()
    
class SparsityAgnosticLassoAgent(Agent):
    def __init__(self, tuningParam: float = 1., d: int = None):
        '''
        args:
            tuningParam: used as lambda for LASSO, defaults to 1
            d: dimensionality passed in from environment
        '''
        super().__init__()
        self.tuningParam = tuningParam
        self.d = d
        self.lt = tuningParam
        self.betaHat = np.zeros(d)
    
    def estimate_rewards(self, context):
        estimates = [np.dot(context[i], self.betaHat) for i in range(len(self.bandits))]
        return estimates
    
    def _get_current_best_bandit(self, context: np.array) -> Bandit:
        estimates = self.estimate_rewards(context)
        return self.bandits[np.argmax(estimates)], np.argmax(estimates)

    def _update_betaHat(self):
        #lasso = GLM(distr = 'gaussian', alpha = 1, reg_lambda = self.lt, max_iter = 1000, fit_intercept = False)
        lasso=linear_model.Lasso(alpha=self.lt)
        X = np.array(self.history_log.all_contexts)
        Y = self.history_log.all_rewards
        lasso.fit(X, Y)
        #return lasso.beta_
        return lasso.coef_


    def _update_tuning_parameter(self):
        t = self.history_log.total_actions
        return self.tuningParam * np.sqrt((4 * np.log(t) + 2 * np.log(self.d))/t)

    def take_action(self, context):
        current_bandit, index = self._get_current_best_bandit(context)
        reward = current_bandit.pull(context[index])
        self.history_log.record_action(current_bandit, context[index], reward)
        self.lt = self._update_tuning_parameter()
        self.betaHat = self._update_betaHat()
    
    def __repr__(self):
        return f'SparsityAgnosticLassoAgent'

class OracleAgent(Agent):
    def __init__(self, true_beta: np.array):
        '''
        args:
            true_beta: true beta of the bandits
        '''
        super().__init__()
        self.true_beta = true_beta
    
    def estimate_rewards(self, context):
        estimates = [np.dot(context[i], self.true_beta) for i in range(len(self.bandits))]
        return estimates
    
    def _get_current_best_bandit(self, context: np.array) -> Bandit:
        estimates = self.estimate_rewards(context)
        return self.bandits[np.argmax(estimates)], np.argmax(estimates)

    def take_action(self, context):
        current_bandit, index = self._get_current_best_bandit(context)
        reward = current_bandit.pull(context[index])
        self.history_log.record_action(current_bandit, context[index], reward)
    
    def __repr__(self):
        return 'OracleAgent'

class DRLassoAgent(Agent):
    #code adapted from https://github.com/gisoo1989/Doubly-Robust-Lasso-Bandit/blob/master/DR%20Lasso_%20main%20simulation.py
    def __init__(self,lam1,lam2,d,N,tc,tr,zt):
        '''
        Args:
            lam1: tuning parameter
            lam2: tuning parameter
            d: dimensionality passed from environment
            N: number of arms
            tc: 
            tr:
            zt: tuning parameter
        '''
        super().__init__()
        self.x=[]
        self.r=[]
        self.lam1=lam1
        self.lam2=lam2
        self.d=d
        self.N=N
        self.beta=np.zeros(d)
        self.tc=tc
        self.tr=tr
        self.zt=zt
        
    def choose_a(self, t, x):  # x is N*d matrix
        '''
        Args:
            t: current iteration
            x: context vectors
        '''
        if t<self.zt:
            self.action=np.random.choice(range(self.N))
            self.pi=1./self.N
        else:
            uniformp=self.lam1*np.sqrt((np.log(t)+np.log(self.d))/t)
            #print(uniformp)
            uniformp=np.minimum(1.0,np.maximum(0.,uniformp))
            choice=np.random.choice([0,1],p=[1.-uniformp,uniformp])
            est=np.dot(x,self.beta)
            if choice==1:
                self.action=np.random.choice(range(self.N))
                if self.action==np.argmax(est):
                    self.pi=uniformp/self.N+(1.-uniformp)
                else:
                    self.pi=uniformp/self.N            
            else:
                self.action=np.argmax(est)
                self.pi=uniformp/self.N+(1.-uniformp)
        #print(self.pi)
        self.x.append(np.mean(x,axis=0))
        #print(np.mean(Xmat,axis=0).shape)
        #print(self.x[-1])
        self.rhat=np.dot(x,self.beta)
        #print(self.rhat)
        return(self.action)            
             
     
    def update_beta(self,rwd,t):
        #print(rwd)
        pseudo_r=np.mean(self.rhat)+(rwd-self.rhat[self.action])/self.pi/self.N
        if self.tr==True:
            pseudo_r=np.minimum(3.,np.maximum(-3.,pseudo_r))
        self.r.append(pseudo_r)
        #print(pseudo_r)
        if t>5:
            if t>self.tc:
                lam2_t=self.lam2*np.sqrt((np.log(t)+np.log(self.d))/t) 
            lasso=linear_model.Lasso(alpha=lam2_t)
            #print(len(self.r))
            lasso.fit(self.x,self.r)
            self.beta=lasso.coef_
    
    def take_action(self, t, context):
        index = self.choose_a(t, context)
        current_bandit = bandits[index]
        reward = current_bandit.pull(context[index])
        self.history_log.record_action(current_bandit, context[index], reward)
        self.update_beta(reward, t)
        
    def __repr__(self):
        return 'DRLassoAgent'

class Environment():
    def __init__(self, dimensions: int, num_arms: int, corr: float) -> None:
        '''
        args:
            dimensions: dimensionality of context data
            num_arms: how many bandits
            corr: correlation between arms
        '''
        self.d = dimensions
        self.k = num_arms
        #create covariance matrix with 1 on diagonal and corr elsewhere
        self.corr = corr * np.ones((self.k, self.k))
        np.fill_diagonal(self.corr, 1)
        self.turns = 0
    
    def generate_context(self):
        context = np.random.multivariate_normal(np.zeros(self.k), cov = self.corr, size = (self.d))
        self.turns += 1
        return context.T

if __name__ == '__main__':
    #initial simulation code 
    np.random.seed(800)
    d = 100
    density = .05
    s0 = d * density
    arms = 20
    corr = 0.7
    testenv = Environment(d, arms, corr)
    beta_true = sps.rand(d, 1, density)
    beta_true = np.array(beta_true.todense())
    bandits = [SGLB(beta_true) for _ in range(arms)]
    test_agent = SparsityAgnosticLassoAgent(d = testenv.d)
    test_agent.bandits = bandits
    oracle_agent = OracleAgent(beta_true)
    oracle_agent.bandits = bandits
    DR_agent = DRLassoAgent(1, 0.5, d, arms, tc = 1, tr = True, zt = 10)
    DR_agent.bandits = bandits
    n = 0
    while testenv.turns < 1000:
        features = testenv.generate_context()
        print(f'turn {testenv.turns}')
        test_agent.take_action(features)
        oracle_agent.take_action(features)
        DR_agent.take_action(testenv.turns, features)
        
    plt.plot(np.cumsum(test_agent.history_log.all_rewards), label=str(test_agent))
    plt.plot(np.cumsum(oracle_agent.history_log.all_rewards), label=str(oracle_agent))
    plt.plot(np.cumsum(DR_agent.history_log.all_rewards), label=str(DR_agent))
    plt.xlabel("iteration")
    plt.ylabel("cumulative rewards")
    plt.title(f"K = {arms}, d = {d}, s0 = {s0}, correlation = {corr}")
    plt.legend()
    plt.show()
    #print(test_agent.betaHat)
    #print(beta_true)
