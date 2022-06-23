import os
from os import path
from mab import Bandits
import numpy as np
import pandas as pd
#import scipy.stats as stats
#from math import sqrt
from scipy.stats import norm,t
from sys import platform
import gc
from statistics import mean 
from numpy.random import choice
import math
from math import log,ceil,sqrt
import matplotlib.pyplot as plt
#import scipy.interpolat as inplt



def lambda_t1(t):
    y=log(t)/t
    return y

def lambda_t2(t):
    y=1.0/log(t)
    return y        

def lambda_t3(t):
    y=1.0/t
    return y

def random_select(n):
    selected = choice(range(n), 1, replace=True)[0]
    return selected

def squared_list(ls):
    return [i**2 for i in ls]

def sum_ls_ij(ls):
    ls = ls[~np.isnan(ls)]
    sum = 0
    for i in ls:
        for j in ls:
            if i!=j:
                sum += i*j
    return sum

def check_sign(ls):
    return (ls[0]*ls[1] > 0) & (ls[0] + ls[1] > 0 )

def compare_list(ls,low,up):
    return [low<i<up for i in ls]


def count_char1(txt):
    result = 0
    for char in txt:
        if char=="1":
            result += 1     
    return result


class Boot_Experiment():
    def __init__(self, x, infer_bool, plot_bool, boot_number, dist, sys_sep, algo, lin, lin_t, lin_const, random_choice,true_prob, sample_size, repeats, gap, gd_eps, sigma, lil_delta, lil_eps, lil_lambda, lil_beta, dp, dp_eps):
        self.x = x
        self.infer_bool = infer_bool
        self.plot_bool = plot_bool
        self.B = boot_number
        self.random_choice = random_choice
        self.algo = algo
        self.lin = lin
        self.lin_t = lin_t
        self.lin_const = lin_const
        self.repeats = repeats
        self.dist = dist
        self.sys_sep = sys_sep
        self.true_prob = true_prob
        self.sample_size = sample_size
        self.gap = gap
        self.bandit_num = len(true_prob)
        self.gd_eps = gd_eps
        self.sigma = sigma
        self.lil_delta = lil_delta
        self.lil_eps = lil_eps
        self.lil_lambda = lil_lambda
        self.lil_beta = lil_beta
        self.dp = dp
        self.dp_eps = dp_eps

    def set_path(self):
        cwdname = os.getcwd()
        if self.dp:
            folder = cwdname + self.sys_sep + self.random_choice+ str(self.sample_size) + self.sys_sep + "experiments_"+ self.algo + "_dp" + str(self.dp_eps) + "_" + self.dist
        elif self.lin:
            folder = cwdname + self.sys_sep + self.random_choice+str(self.sample_size) + self.sys_sep + "experiments_"+ self.algo + "_linear"+"_" + self.dist
        elif self.lin_t:
            folder = cwdname + self.sys_sep + self.random_choice+str(self.sample_size) + self.sys_sep + "experiments_"+ self.algo + "_linear_const"+"_" + self.dist
        else:
            folder = cwdname + self.sys_sep + self.random_choice+str(self.sample_size) + self.sys_sep + "experiments_"+ self.algo + "_original"+"_" + self.dist

        if path.exists(folder)!= 1:
            os.makedirs(folder)
        self.folder = folder


    def boot_experiment(self):
        x = self.x
        infer_bool = self.infer_bool
        bandits = Bandits(random_choice=self.random_choice, true_prob=self.true_prob, sample_size=self.sample_size,
            algo=self.algo, lin=self.lin, lin_t=self.lin_t, lin_const=self.lin_const, gd_eps=self.gd_eps,
             sigma=self.sigma, lil_delta=self.lil_delta, lil_eps=self.lil_eps, lil_lambda=self.lil_lambda, 
             lil_beta=self.lil_beta, dp=self.dp, dp_eps=self.dp_eps)
        (bandit_stats, output_bias, output_means, output_pulls, output_cov, output_corr, output_reward, output_pull_indicator) = bandits.run()

        B = self.B
        K = self.bandit_num
        boot_mix_dict = dict.fromkeys((range(B)))

        output_ecdf = np.zeros((len(x), self.bandit_num))
        output_ecdf_gap = np.zeros((len(x), self.bandit_num))
        output_ecdf_boot = np.zeros((len(x), self.bandit_num))


        boot_stats = np.zeros((self.bandit_num,), dtype=[('true_means',np.float),('true_sigma',np.float),
            ('mab_means', np.float),('boot_means',np.float),
            ('mab_pulls',np.float),('boot_pulls', np.float),
            ('cov', np.float), ('avg_means', np.float),
            ('gap_bias', np.float),('boot_bias', np.float),
            ('var_reward', np.float),('var_mabmean', np.float),('var_reward_avg', np.float),
            ('zstat_mab', np.float),('zstat_mab_mark', bool),
            ('zstat_thm2', np.float),('zstat_thm2_mark', bool),
            ('zstat_thm3', np.float),('zstat_thm3_mark', bool),
            ('zstat_thm23', np.float),('zstat_thm23_mark', bool),
            ('var_correct_thm2', np.float),
            ('var_correct_thm3', np.float),
            ('var_correct_thm23', np.float)])
        
        boot_stats['true_sigma'] = self.sigma
        boot_stats['true_means'] = self.true_prob
        boot_stats['mab_means']=bandit_stats['means']
        boot_stats['var_reward'] = np.nanvar(output_reward, axis=0)
        boot_stats['mab_pulls'] = bandit_stats['pulls']
        boot_stats['var_reward_avg'] = np.nanvar(output_reward, axis=0)/boot_stats['mab_pulls']


        ######################################################################################################################################
        for b in range(B):
            boot_mix_dict[b] = dict.fromkeys((range(K)))
            boot_Bandits = Boot_Bandits(output_reward=output_reward, random_choice=self.random_choice, 
                true_prob=self.true_prob, sample_size=self.sample_size, algo=self.algo, 
                lin=self.lin, lin_t=self.lin_t, lin_const=self.lin_const, gd_eps=self.gd_eps, 
                sigma=self.sigma, lil_delta=self.lil_delta, lil_eps=self.lil_eps, 
                lil_lambda=self.lil_lambda, lil_beta=self.lil_beta, dp=self.dp, dp_eps=self.dp_eps)
            (boot_result, output_reward_boot) = boot_Bandits.boot_run()

            for k in range(K):
                reward_k = output_reward_boot[:, k]
                reward_k = reward_k[~np.isnan(reward_k)]
                boot_mix_dict[b][k] = {"mean": boot_result['means'][k], "pull": boot_result['pulls'][k],
                                       "ecdf":dict.fromkeys(x+self.true_prob[k]),
                                       "squared_rewards": sum(squared_list(reward_k)),
                                       "sum_diff_rewards": sum_ls_ij(reward_k)}
                for i,xx in enumerate(x+self.true_prob[k]):
                    boot_mix_dict[b][k]['ecdf'][xx] = np.sum(reward_k<=xx)/float(len(reward_k))
                    #output_ecdf[i,k] = mix_dict[b][k]['ecdf'][xx]


        for k in range(K):
            mean_list = []
            pull_list = []
            squared_rewards_list = []
            sum_diff_rewards_list = []
            ecdf_np=np.zeros((B, len(x)))
            reward_k = output_reward[:, k]
            reward_k = reward_k[~np.isnan(reward_k)]

            for b in range(B):
                mean_list.append(boot_mix_dict[b][k]['mean'])
                pull_list.append(boot_mix_dict[b][k]['pull'])
                squared_rewards_list.append(boot_mix_dict[b][k]['squared_rewards'])
                sum_diff_rewards_list.append(boot_mix_dict[b][k]['sum_diff_rewards'])

                for i, xx in enumerate(x+self.true_prob[k]):
                    ecdf_np[b,i] = boot_mix_dict[b][k]['ecdf'][xx]
            
            
            mean_list = np.array(mean_list)
            pull_list = np.array(pull_list)
            squared_rewards_list = np.array(squared_rewards_list)
            sum_diff_rewards_list = np.array(sum_diff_rewards_list)            

            for i, xx in enumerate(x+self.true_prob[k]):
                output_ecdf[i, k] = np.sum(reward_k <= xx) / float(len(reward_k))
                output_ecdf_gap[i, k] = np.cov(ecdf_np[:,i],pull_list)[1][0]/mean(pull_list)
                output_ecdf_boot[i, k] = output_ecdf[i, k] + output_ecdf_gap[i, k] if (output_ecdf[i, k] + output_ecdf_gap[i, k]) <=1 else 1

            boot_stats['avg_means'][k]=mean(mean_list)
            boot_stats['boot_pulls'][k]=mean(pull_list)
            boot_stats['cov'][k]=np.cov(mean_list,pull_list)[1][0]
            boot_stats['gap_bias'][k] = float(boot_stats['cov'][k]/boot_stats['boot_pulls'][k])
            boot_stats['boot_means'][k] = boot_stats['mab_means'][k]+boot_stats['gap_bias'][k]
            boot_stats['boot_bias'][k] = boot_stats['boot_means'][k]-self.true_prob[k]



            sample_variance = boot_stats['var_reward'][k]
            ##print(squared_rewards_list)
            #print(np.cov(squared_rewards_list/pull_list, pull_list))
            sample_cov1 = np.cov(squared_rewards_list/pull_list, pull_list)[1][0]/mean(pull_list)
            sample_cov2 = np.cov(sum_diff_rewards_list/pull_list/(pull_list-1),pull_list*(pull_list-1))[1][0]/mean(pull_list*(pull_list-1))
            sample_variance_est = sample_variance + sample_cov1 + sample_cov2
            
            var_mabmean2 = np.cov(squared_list(mean_list),squared_list(pull_list))[1][0]/mean(squared_list(pull_list))
            var_mabmean3 = boot_stats['gap_bias'][k]**2
            var_mabmean4 = 2*boot_stats['boot_means'][k]*boot_stats['gap_bias'][k]
            
            # if var_correction = 'sample_variance':
            var_mean_correct_thm2 = sample_variance_est/boot_stats['boot_pulls'][k]
            boot_stats['var_correct_thm2'][k] = var_mean_correct_thm2
                
            #elif var_correction = 'var_mabmean':
            var_mabmean1 = mean(pull_list)/mean(squared_list(pull_list))*sample_variance
            var_mean_correct_thm3 = var_mabmean1 - var_mabmean2 - var_mabmean3 + var_mabmean4
            boot_stats['var_correct_thm3'][k] = var_mean_correct_thm3
            #else both correct
            var_mabmean1_new = mean(pull_list)/mean(squared_list(pull_list))*sample_variance_est
            var_mean_correct_thm23 = var_mabmean1_new - var_mabmean2 - var_mabmean3 + var_mabmean4
            boot_stats['var_correct_thm23'][k] = var_mean_correct_thm23
                

            #boot_stats['var_mabmean'][k] = var_mabmean1 - var_mabmean2 - var_mabmean3 + var_mabmean4
    
        if infer_bool:
            #H0: mu1-mu2 > 0.5 vs H1: mu1-mu2 < 0.5
            numerator = (boot_stats['mab_means'][0]-boot_stats['mab_means'][1])-(boot_stats['true_means'][0]-boot_stats['true_means'][1])
            boot_stats['zstat_mab'][0] = numerator/sqrt(boot_stats['var_reward'][0]/boot_stats['mab_pulls'][0]+boot_stats['var_reward'][1]/boot_stats['mab_pulls'][1])
            boot_stats['zstat_mab_mark'][0] = (boot_stats['zstat_mab'][0] < -1.65)
            
            if check_sign(boot_stats['var_correct_thm2']):
                boot_stats['zstat_thm2'][0] = numerator/sqrt(boot_stats['var_correct_thm2'][0]+boot_stats['var_correct_thm2'][1])
                boot_stats['zstat_thm2_mark'][0] = (boot_stats['zstat_thm2'][0] < -1.65)
            else:
                boot_stats['zstat_thm2_mark'][0] = np.nan
                
            if check_sign(boot_stats['var_correct_thm3']):
                boot_stats['zstat_thm3'][0] = numerator/sqrt(boot_stats['var_correct_thm3'][0]+boot_stats['var_correct_thm3'][1])
                boot_stats['zstat_thm3_mark'][0] = (boot_stats['zstat_thm3'][0] < -1.65)
            else:
                boot_stats['zstat_thm3_mark'][0] = np.nan            
            
            if check_sign(boot_stats['var_correct_thm23']):
                boot_stats['zstat_thm23'][0] = numerator/sqrt(boot_stats['var_correct_thm23'][0]+boot_stats['var_correct_thm23'][1])
                boot_stats['zstat_thm23_mark'][0] = (boot_stats['zstat_thm23'][0] < -1.65)
            else:
                boot_stats['zstat_thm23_mark'][0] = np.nan
        
        else:
            pass
            



        if self.plot_bool:
            plt.figure()
            fig, axes = plt.subplots(self.bandit_num, sharex=True, sharey=True)
            for k in range(self.bandit_num):
                axes[k].plot(x+self.true_prob[k], norm.cdf(x+self.true_prob[k], loc=self.true_prob[k], scale=self.sigma), label='cdf')
                axes[k].plot(x+self.true_prob[k], output_ecdf[:,k],  label='ecdf_naive')
                axes[k].plot(x+self.true_prob[k], output_ecdf_boot[:,k],  label='ecdf_boot')
                axes[k].legend(loc='upper left')
            plt.show()
        else:
            pass



        return boot_stats, output_ecdf, output_ecdf_gap, output_ecdf_boot

    def boot_output(self):
        boot_stats = self.boot_experiment()
        boot_stats_ot = pd.DataFrame(boot_stats)
        boot_stats_ot.to_csv(self.folder + self.sys_sep + "boot_bias" + ".csv")



class Boot_Bandits():
    def __init__(self, output_reward, random_choice, true_prob, sample_size, algo, lin, lin_t, lin_const, gd_eps, sigma, lil_delta, lil_eps, lil_lambda, lil_beta, dp, dp_eps):
        self.output_reward = output_reward
        self.random_choice = random_choice
        self.true_prob = true_prob
        self.bandit_num = len(true_prob)
        self.max_prob = max(true_prob)
        self.sample_size = sample_size
        self.algo = algo
        self.lin = lin
        self.lin_t = lin_t
        self.lin_const = lin_const
        self.gd_eps = gd_eps
        self.sigma = sigma
        self.lil_delta = lil_delta
        self.lil_eps = lil_eps
        self.lil_lambda = lil_lambda
        self.lil_beta = lil_beta
        self.dp = dp
        self.dp_eps = dp_eps
        self.bandit_stats = np.zeros((self.bandit_num,), dtype=[('total_rewards', np.float),
            ('pulls', np.int), ('means', np.float), ('bias', np.float),
            ('sum_sqr_rewards', np.float), ('covariance', np.float),('corr', np.float)])
                # +1 means the last column is regret
        #self.output_bias = np.zeros((self.sample_size, self.bandit_num+1))
        #self.output_pulls = np.zeros((self.sample_size, self.bandit_num))
        #self.output_cov = np.zeros((self.sample_size, self.bandit_num))
        #self.output_means = np.zeros((self.sample_size, self.bandit_num))
        #self.output_means[:] = np.NaN
        #self.output_corr = np.zeros((self.sample_size, self.bandit_num))
        #self.output_exp_bias = np.zeros((self.sample_size, self.bandit_num))
        self.output_reward_boot = np.empty((self.sample_size, self.bandit_num))
        self.output_reward_boot[:] = np.NaN
        #self.output_pull_indicator = np.zeros((self.sample_size, self.bandit_num))


        if self.bandit_num < 2:
            raise ValueError('Number of bandits should be greater than 1')




    def epsilon_greedy_select(self,t):
        if np.random.binomial(1, self.gd_eps):
            #exploration:
            selected = random_select(self.bandit_num)
        else:
            #exploitation:
            if self.dp:
                selected = (self.bandit_stats['means']+self.dp_noise(t)/self.bandit_stats['pulls']).argmax()
            else:
                selected = self.bandit_stats['means'].argmax()
        return selected


    def t_greedy_select(self,t):
        if np.random.binomial(1, lambda_t3(t)):
            #exploration:
            selected = random_select(self.bandit_num)
        else:
            #exploitation:
            if self.dp:
                selected = (self.bandit_stats['means']+self.dp_noise(t)/self.bandit_stats['pulls']).argmax()
            else:
                selected = self.bandit_stats['means'].argmax()
        return selected


    def greedy_select(self,t):
        if self.dp:
            selected = (self.bandit_stats['means']+self.dp_noise(t)/self.bandit_stats['pulls']).argmax()
        else:
            selected = self.bandit_stats['means'].argmax()        
        return selected

    def ucb_select(self, t):
        max_upper_bound = 0
        selected = 0
        for k in range(self.bandit_num):
            delta_k = math.sqrt(2*math.log(t+1)/self.bandit_stats['pulls'][k])
            if self.dp:
                mean = self.bandit_stats['means'][k]+self.dp_noise(t)/self.bandit_stats['pulls'][k]
            else:
                mean = self.bandit_stats['means'][k]
            upper_bound = mean + delta_k
            if (upper_bound > max_upper_bound):
                max_upper_bound = upper_bound
                selected = k
        return selected

    def ucb1_normal_select(self, t):
        max_upper_bound = 0
        selected = 0
        for k in range(self.bandit_num):
            if self.bandit_stats['pulls'][k] < ceil(8*log(t+1)):
                selected = k
                break
        if np.min(self.bandit_stats['pulls']) >= ceil(8*log(t+1)):
            for k in range(self.bandit_num):
                n = self.bandit_stats['pulls'][k]
                if self.dp:
                    mean = self.bandit_stats['means'][k]+self.dp_noise(t)/self.bandit_stats['pulls'][k]
                else:
                    mean = self.bandit_stats['means'][k]
                sum_sqr = self.bandit_stats['sum_sqr_rewards'][k]
                delta_k = 4*math.sqrt(sum_sqr-n*(mean**2))*log(t)/n/(n-1)
                upper_bound = mean + delta_k
                if (upper_bound > max_upper_bound):
                    max_upper_bound = upper_bound
                    selected = k
        return selected

    def lil_ucb_select(self, t):
        max_upper_bound = 0
        selected = 0
        for k in range(self.bandit_num):
            n = self.bandit_stats['pulls'][k]
            coeff = (1+self.lil_beta)*(1+sqrt(self.lil_eps))
            delta_k = coeff*sqrt(2*self.sigma**2*(1+self.lil_eps)*log(log((1+self.lil_eps)*n)/self.lil_delta)/n)
            if self.dp:
                mean = self.bandit_stats['means'][k]+self.dp_noise(t)/self.bandit_stats['pulls'][k]
            else:
                mean = self.bandit_stats['means'][k]
            upper_bound = mean + delta_k
            if (upper_bound > max_upper_bound):
                max_upper_bound = upper_bound
                selected = k
        return selected

    def ts_select(self, t):
        thetamax = 0
        selected = 0
        for k in range(self.bandit_num):
            if self.dp:
                mean = self.bandit_stats['means'][k]+self.dp_noise(t)/self.bandit_stats['pulls'][k]
            else:
                mean = self.bandit_stats['means'][k]            
            rnd = np.random.normal(loc=mean, scale=1/(self.bandit_stats['pulls'][k]+1), size = 1)
            if rnd >= thetamax:
                thetamax = rnd
                selected = k
        return selected

    def mab_select(self, t):
        if self.algo == "ucb":
            selected = self.ucb_select(t)
        elif self.algo == "ucb1_normal":
            selected = self.ucb1_normal_select(t)
        elif self.algo =="lil_ucb":
            selected = self.lil_ucb_select(t)
        elif self.algo == "ts":
            selected = self.ts_select(t)
        elif self.algo == "epsilon_greedy":
            selected = self.epsilon_greedy_select(t)
        elif self.algo == "greedy":
            selected = self.greedy_select(t)
        elif self.algo == "t_greedy":
            selected = self.t_greedy_select(t)
        elif self.algo == "equal":
            selected = random_select(self.bandit_num)
        return selected

    def select(self, t):
        #if self.lin_const > self.bandit_num:
        #    raise ValueError('Number of lin_const should be smaller than bandit number')

        if t < self.bandit_num:
            selected = t
        else:
            u = np.random.uniform(low=0.0, high=1.0, size=1)
            if self.lin:
                lambda_t = lambda_t3(t+1)
            elif self.lin_t:
                lambda_t = self.lin_const*lambda_t3(t+1)
            else:
                lambda_t = 0

            if u < lambda_t:
                if self.random_choice == "uniform":
                    selected = random_select(self.bandit_num)
                else: 
                #   self.random_choice == "water_filling":
                    selected = np.argmin(self.bandit_stats['pulls'])
            else:
                selected = self.mab_select(t)
        return selected

    def boot_update(self, t, selected):
        reward_K = self.output_reward[:,selected]
        reward_K = reward_K[~np.isnan(reward_K)]
        temp_reward = np.random.choice(reward_K, size=1, replace=True)[0]

        self.output_reward_boot[t][selected] = temp_reward
        self.bandit_stats['total_rewards'][selected] = self.bandit_stats['total_rewards'][selected] +  temp_reward
        self.bandit_stats['pulls'][selected] = self.bandit_stats['pulls'][selected] + 1 
        self.bandit_stats['means'][selected] = self.bandit_stats['total_rewards'][selected]/self.bandit_stats['pulls'][selected]
        self.bandit_stats['bias'][selected] = self.bandit_stats['means'][selected]-self.true_prob[selected]        


    def boot_run(self):
        for t in range(self.sample_size):
            selected = self.select(t)
            self.boot_update(t, selected)
        return self.bandit_stats, self.output_reward_boot


    def dp_noise(self,t):
        t_bin_cnt=0
        n_digit = int(ceil(log(self.sample_size,2)))
        dp_eps0 = self.dp_eps/self.bandit_num

        #convert t into binary representation
        t_bin=bin(t)[2:].zfill(n_digit)
        t_bin_cnt = count_char1(t_bin)
        sum_lap = np.random.laplace(loc=0.0, scale=t_bin_cnt/dp_eps0)
        return sum_lap







