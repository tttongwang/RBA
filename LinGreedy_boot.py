# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 03:25:46 2022

@author: tongw
"""

from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
from math import log, ceil, sqrt
from numpy.random import choice
#from scipy.stats import norm,t


def random_select(n):
    selected = choice(range(n), 1, p=np.repeat(1/n, n))[0]
    return selected

def count_char1(txt):
    result = 0
    for char in txt:
        if char=="1":
            result += 1
    return result

def d_unif_unit(d):
    """
    A function used to define a d-dim vector which is uniformly distributed
    on the surface of unit ball 

    Attributes
    ----------
    d : int

    """
    d_unif = np.random.uniform(low=0.0, high=1.0, size=d)
    sum_sqr = np.sum(np.square(d_unif))
    d_unif_unit = d_unif/float(np.sqrt(sum_sqr))
    return d_unif_unit



def get_context(d):
    '''
        assume xi~bern(p)
        
        Para:
        -----
        d: dim of context features
        p: bern para
        
        Return:
        -------
        a random vector that represents the contextual information
        1 means intercept term
    '''
    return np.float32(np.random.binomial(size=d,n=1,p=0.5))
        
    
    
    
    
class LinGreedy_boot():
    def __init__(self, theta, linucb_alpha, feature_d, K, sample_size, dd_total ):
        self.linucb_alpha = linucb_alpha # confidence constant in Lin_UCB
        self.bandit_num = K
        self.sample_size = sample_size
        self.d = feature_d #d: dimension of user features
        self.dd_total = dd_total # it stores X and y for all arms


        self.output_reward = np.empty((self.sample_size, self.bandit_num))
        self.output_reward[:] = np.NaN
        self.pa = np.zeros((self.sample_size, self.bandit_num))

        self.Aa = dict.fromkeys(range(K)) #Aa: collection of matrix to compute disjoint part for each arm
        self.AaI = dict.fromkeys(range(K)) #AaI: store the inverse for all Aa matrix
        self.ba = dict.fromkeys(range(K)) #ba: collection of vectors to compute disjoint parts
        self.xaT = dict.fromkeys(range(K))
        self.xa = dict.fromkeys(range(K))
        self.X = dict.fromkeys(range(K)) #X: store design matrix

        self.est_theta = dict.fromkeys(range(K))

        #self.theta = dict.fromkeys(range(K)) #theta: store the true theta for all K arms
        self.theta = theta
        
        
        # initialize settings at time 0
        for key in range(K):
            self.Aa[key] = np.identity(self.d) #create the identity matrix
            self.ba[key] = np.zeros(self.d)
            self.AaI[key] = np.identity(self.d)
            self.est_theta[key] = np.zeros(self.d)
            self.X[key] = np.empty((self.sample_size,self.d))
            self.X[key][:] = np.NaN
            #self.theta[key] = d_unif_unit(self.d)

        # warning settings
        if self.bandit_num < 2:
            raise ValueError('Number of bandits should be greater than 1')


    def select(self,t):
        for key in range(self.bandit_num):
            self.est_theta[key] = np.dot(np.linalg.inv(self.Aa[key]), self.ba[key])
            self.xa[key] = get_context(self.d)
            self.xaT[key] = np.transpose(self.xa[key])
            #std = np.sqrt(np.dot(np.dot(self.xaT[key],self.AaI[key]),self.xa[key]))
            #std = np.sqrt(np.linalg.multi_dot([self.xaT[key],np.linalg.inv(self.Aa[key]),self.xa[key]]))
            self.pa[t,key] = np.dot(self.xaT[key],self.est_theta[key])
        if t < self.bandit_num:
            selected = t
        else:
            #selected = np.argmax(self.pa[t,])
            selected = np.random.choice(np.flatnonzero(self.pa[t,] == self.pa[t,].max()))

        return selected


    def update(self, selected, t):
        #print(self.dd_total[selected][str(self.xa[selected])])
        #temp_reward = np.random.normal(loc=np.dot(self.xaT[selected], self.theta[selected]),scale=1)
        #print('t:',t)
        #print('selected',selected)
        #print(self.dd_total[selected][str(self.xa[selected])])
        if isinstance(self.dd_total[selected][str(self.xa[selected])], list):
            #print('True2')
            temp_reward = np.random.choice(self.dd_total[selected][str(self.xa[selected])],1)
        else:
            #print(self.dd_total[selected][str(self.xa[selected])])
            temp_reward = self.dd_total[selected][str(self.xa[selected])]
   
        self.output_reward[t,selected] = temp_reward

        self.Aa[selected] += np.dot(np.reshape(self.xa[selected],(self.d,1)), np.reshape(self.xa[selected],(1,self.d)))
        #self.AaI[selected] = np.linalg.inv(self.Aa[selected])
        self.ba[selected] = self.ba[selected] + temp_reward * self.xa[selected]
        self.X[selected][t,:] = self.xa[selected]
        #self.est_theta[selected] = np.dot(self.AaI[selected], self.ba[selected])



    def run(self):
        for t in range(self.sample_size):
            selected = self.select(t)
            self.update(selected,t)
        for k in range(self.bandit_num):
            self.X[k]= self.X[k][~np.isnan(self.X[k]).all(axis=1)]
        return self.est_theta, self.output_reward, self.ba, self.X


