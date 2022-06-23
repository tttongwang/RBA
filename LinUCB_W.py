from __future__ import division
import numpy as np


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
    return np.random.binomial(size=d,n=1,p=0.5)
    #return np.random.normal(loc=0, scale=1,size=d)
        
    
    
    
    
class LinUCB_W():
    def __init__(self, theta, linucb_alpha, feature_d, K, sample_size, fixed_lambda):
        self.linucb_alpha = linucb_alpha # confidence constant in Lin_UCB
        self.bandit_num = K
        self.sample_size = sample_size
        self.d = feature_d #d: dimension of user features


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
        self.est_theta_W = dict.fromkeys(range(K))
        
        #self.theta = dict.fromkeys(range(K)) #theta: store the true theta for all K arms
        self.theta = theta
        
        # inital settings for W estimator
        self.wi_deno = np.ones(K)
        self.WX = dict.fromkeys(range(K))
        self.W = dict.fromkeys(range(K))
        self.N = np.zeros(K, dtype='int')
        self.fixed_lambda = fixed_lambda
        # initialize settings at time 0
        for key in range(K):
            self.Aa[key] = np.identity(self.d) #create the identity matrix
            self.ba[key] = np.zeros(self.d)
            self.AaI[key] = np.identity(self.d)
            self.est_theta[key] = np.zeros(self.d)
            self.X[key] = np.empty((self.sample_size,self.d))
            self.X[key][:] = np.NaN
            #self.theta[key] = d_unif_unit(self.d)
            #self.WX[key] = np.zeros((self.d,self.d))
            self.WX[key] = np.zeros((self.d,self.d))
            self.W[key] = np.zeros((self.d,1))
            #self.W[key] = 0
        # warning settings
        if self.bandit_num < 2:
            raise ValueError('Number of bandits should be greater than 1')


    def select(self,t):
        for key in range(self.bandit_num):
            self.est_theta[key] = np.dot(np.linalg.inv(self.Aa[key]), self.ba[key])
            self.xa[key] = get_context(self.d)
            self.xaT[key] = np.transpose(self.xa[key])
            #std = np.sqrt(np.dot(np.dot(self.xaT[key],self.AaI[key]),self.xa[key]))
            std = np.sqrt(np.linalg.multi_dot([self.xaT[key],np.linalg.inv(self.Aa[key]),self.xa[key]]))
            self.pa[t,key] = np.dot(self.xaT[key],self.est_theta[key])+self.linucb_alpha*std
        if t < self.bandit_num:
            selected = t
        else:
            #selected = np.argmax(self.pa[t,])
            selected = np.random.choice(np.flatnonzero(self.pa[t,] == self.pa[t,].max()))

        return selected


    def update(self, selected, t):
        temp_reward = np.random.normal(loc=np.dot(self.xaT[selected], self.theta[selected]),scale=1)
        #temp_reward = np.random.normal(loc=np.dot(np.square(self.xaT[selected]), self.theta[selected]),scale=1)

        self.output_reward[t,selected] = temp_reward
        self.Aa[selected] += np.dot(np.reshape(self.xa[selected],(self.d,1)), np.reshape(self.xa[selected],(1,self.d)))
        #self.AaI[selected] = np.linalg.inv(self.Aa[selected])
        self.ba[selected] = self.ba[selected] + temp_reward * self.xa[selected]
        self.X[selected][self.N[selected],:] = self.xa[selected]
        #self.est_theta[selected] = np.dot(self.AaI[selected], self.ba[selected])
        self.N[selected] += 1


    def update_W(self, W_lambda_list):
        for k in range(self.bandit_num):
            for i in range(self.N[k]):
                wi_deno = W_lambda_list[k] + np.linalg.norm(self.X[k][i,:])**2
                wi = np.dot((np.identity(self.d) - self.WX[k]),np.reshape(self.X[k][i,:],(self.d,1)))/wi_deno
                self.WX[k] = np.dot(self.W[k], self.X[k][0:i+1,:])
                self.W[k] = np.c_[self.W[k],wi]
        return self.W

        
        
        
    def run(self):
        for t in range(self.sample_size):
            selected = self.select(t)
            self.update(selected,t)
        W_lambda_list = np.zeros(self.bandit_num)
        # get W_lambda_list
        if self.fixed_lambda:
            for k in range(self.bandit_num):
                self.X[k] = self.X[k][~np.isnan(self.X[k]).all(axis=1)]
                W_lambda_list[k] = self.fixed_lambda[k]
            #W_lambda_list = np.repeat(self.fixed_lambda, self.bandit_num)
        else:
            
            for k in range(self.bandit_num):
                self.X[k] = self.X[k][~np.isnan(self.X[k]).all(axis=1)]
                # compute W_lambda
                H = np.dot(self.X[k].T, self.X[k])
                lambda_list, v = np.linalg.eig(H)
                W_lambda_list[k] = np.min(lambda_list)*0.1
                #print('suggest_lambda:', W_lambda_list[k])
        # get Wn
        W = self.update_W(W_lambda_list=W_lambda_list)
        for k in range(self.bandit_num):
            # remove 1st col in W 
            W[k] = np.delete(W[k], 0, 1)
            # remove nan in y
            yk = self.output_reward[:,k][~np.isnan(self.output_reward[:,k])]
            # insert formula
            self.est_theta_W[k] = self.est_theta[k] + np.dot(W[k], yk-np.dot(self.X[k], self.est_theta[k]))
             
        return self.est_theta, self.output_reward, self.ba, self.X, self.est_theta_W


