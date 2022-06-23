from os import path
from mab import Bandits
#import mab
import os
import numpy as np
import pandas as pd
#import scipy.stats as stats
from math import sqrt
from scipy.stats import norm,t
from sys import platform
import gc
from boot_mab2_new import Boot_Experiment, Boot_Bandits
import matplotlib.pyplot as plt



class Experiments():
    def __init__(self, x, infer_bool,plot_total_bool, plot_bool, boot_number, dist, sys_sep, algo, lin, lin_t, lin_const, random_choice,true_prob, sample_size, repeats, gap, gd_eps, sigma, lil_delta, lil_eps, lil_lambda, lil_beta, dp, dp_eps):
        self.x = x
        #self.x = np.arange(-2, 2, 0.25)
        self.infer_bool = infer_bool
        self.plot_total_bool = plot_total_bool
        self.plot_bool = plot_bool
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
        self.boot_number = boot_number

    def set_path(self):
        cwdname = os.getcwd()
        if self.dp:
            folder = cwdname + self.sys_sep + self.random_choice+ str(self.sample_size) + self.sys_sep + "boot_experiments_"+ self.algo + "_dp" + str(self.dp_eps) + "_" + self.dist
        elif self.lin:
            folder = cwdname + self.sys_sep + self.random_choice+str(self.sample_size) + self.sys_sep + "boot_experiments_"+ self.algo + "_linear"+"_" + self.dist
        elif self.lin_t:
            folder = cwdname + self.sys_sep + self.random_choice+str(self.sample_size) + self.sys_sep + "boot_experiments_"+ self.algo + "_linear_const"+"_" + self.dist
        else:
            folder = cwdname + self.sys_sep + self.random_choice+str(self.sample_size) + self.sys_sep + "boot_experiments_"+ self.algo + "_original"+"_" + self.dist
        
        if path.exists(folder)!= 1:
            os.makedirs(folder)
        self.folder = folder


    def run_experiments(self):
        #for the storage of output of single experiment, which involves the result of each step
        mix_dict2 = dict.fromkeys((range(self.repeats)))
        mix_np_ecdf = np.zeros((len(self.x), self.bandit_num, self.repeats))
        mix_np_ecdf_gap = np.zeros((len(self.x), self.bandit_num, self.repeats))
        mix_np_ecdf_boot = np.zeros((len(self.x), self.bandit_num, self.repeats))
        mab_mean_np = np.zeros((self.bandit_num, self.repeats))
        mab_pull_np = np.zeros((self.bandit_num, self.repeats))
        for j in range(self.repeats):
            boot_experiment = Boot_Experiment(x = self.x, infer_bool=self.infer_bool, plot_bool=self.plot_bool, 
                boot_number=self.boot_number, dist = self.dist, sys_sep=self.sys_sep, algo=self.algo,
                lin=self.lin, lin_t=self.lin_t, lin_const=self.lin_const, random_choice=self.random_choice,
                true_prob=self.true_prob, sample_size=self.sample_size, repeats=self.repeats, gap=self.gap, 
                gd_eps=self.gd_eps, sigma=self.sigma, lil_delta=self.lil_delta, lil_eps=self.lil_eps, 
                lil_lambda=self.lil_lambda, lil_beta=self.lil_beta, dp=self.dp, dp_eps=self.dp_eps)

            (boot_stats, output_ecdf, output_ecdf_gap, output_ecdf_boot) = boot_experiment.boot_experiment()
            mix_dict2[j] = boot_stats
            mix_np_ecdf[:,:,j] = output_ecdf
            mix_np_ecdf_gap[:,:,j] = output_ecdf_gap
            mix_np_ecdf_boot[:,:,j] = output_ecdf_boot
            #mab_mean_np[:,j] = boot_stats['mab_means']
            #mab_pull_np[:,j] = boot_stats['mab_pulls']
            cutoff = 0.1

        # a is a collection for boot_stats
        a=np.array(list(mix_dict2.values()))

        return a, mix_np_ecdf, mix_np_ecdf_gap, mix_np_ecdf_boot

    def compute_total_stats(self):
        (a, mix_np_ecdf, mix_np_ecdf_gap, mix_np_ecdf_boot) = self.run_experiments()
        total_stats = np.zeros((self.bandit_num,), dtype=[('true_means',np.float),('true_sigma', np.float),
                                                   ('mab_means', np.float),('boot_means', np.float),
                                                   ('mab_bias', np.float),('boot_bias', np.float),
                                                   ('mab_pulls',np.float),('boot_pulls', np.float),
                                                   #('mab_var', np.float),('boot_var', np.float),
                                                   ('mab_mean_var', np.float), ('boot_mean_var', np.float),
                                                   ('var_reward_avg', np.float),('var_mabmean', np.float),
                                                   #('experiments_oracle_bias_correction', np.float),
                                                   #('experiments_boot_bias_correction', np.float),
                                                   #('avg_boot_bias_correction', np.float),
                                                   #('avg_boot_bias_cov', np.float),
                                                   #('boot_cp', np.float),('mab_cp', np.float),('test_cp', np.float),
                                                   #('mab_corrected_pvalue',np.float),
                                                   #('mab_regular_pvalue',np.float),
                                                   ('mab_pvalue',np.float),
                                                   ('correct_thm2_pvalue',np.float),
                                                   ('correct_thm3_pvalue',np.float),
                                                   ('correct_thm23_pvalue',np.float)])
        total_stats['true_means'] = self.true_prob
        total_stats['true_sigma'] = self.sigma

        total_stats['mab_means'] = np.mean(a['mab_means'], axis=0)
        total_stats['mab_mean_var'] = np.var(a['mab_means'], axis=0)
        total_stats['mab_bias'] = total_stats['mab_means'] - total_stats['true_means']
        total_stats['mab_pulls'] = np.mean(a['mab_pulls'], axis=0)
        #total_stats['mab_var'] = np.mean(a['mab_var'],axis=0)
        total_stats['var_reward_avg'] = np.mean(a['var_reward_avg'], axis=0)
        total_stats['var_mabmean'] = np.mean(a['var_mabmean'], axis=0)

        total_stats['boot_means'] = np.mean(a['boot_means'], axis=0)
        total_stats['boot_mean_var'] = np.var(a['boot_means'], axis=0)
        total_stats['boot_bias'] = total_stats['boot_means'] - total_stats['true_means']
        total_stats['boot_pulls'] = np.mean(a['boot_pulls'], axis=0)
        #total_stats['mab_corrected_pvalue'] = np.mean(a['zstat_mab_mark'],axis=0)
        #total_stats['mab_regular_pvalue'] = np.mean(a['zstat_regular_mark'],axis=0)
        if self.infer_bool:
            total_stats['mab_pvalue'] = np.nanmean(a['zstat_mab_mark'][:, 0])
            total_stats['correct_thm2_pvalue'] = np.nanmean(a['zstat_thm2_mark'][:, 0])
            total_stats['correct_thm3_pvalue'] = np.nanmean(a['zstat_thm3_mark'][:, 0])
            total_stats['correct_thm23_pvalue'] = np.nanmean(a['zstat_thm23_mark'][:, 0])
        else:
            pass


#        total_stats['avg_boot_bias_correction'] = np.mean(a['gap_bias'],axis=0)
#        total_stats['avg_boot_bias_cov'] = np.mean(a['cov'],axis=0)



        #total_stats['boot_cp'] = np.mean(a['boot_ztest_mark'],axis=0)
        #total_stats['mab_cp'] = np.mean(a['mab_ztest_mark'],axis=0)
        #total_stats['test_cp'] = np.mean(a['test_ztest_mark'],axis=0)




        ecdf_mean = mix_np_ecdf.mean(axis=2)
        ecdf_gap_mean = mix_np_ecdf_gap.mean(axis=2)
        ecdf_boot_mean = mix_np_ecdf_boot.mean(axis=2)
        if self.plot_total_bool:
            plt.figure()
            fig, axes = plt.subplots(1,self.bandit_num, sharex=False, sharey=True,figsize=(3.54*5,3.54), dpi=600)
            for k in range(self.bandit_num):
                axes[k].plot(self.x+self.true_prob[k], ecdf_mean[:,k], label='ecdf_naive')
                axes[k].plot(self.x+self.true_prob[k], ecdf_boot_mean[:,k], label='ecdf_boot')
                axes[k].plot(self.x+self.true_prob[k], norm.cdf(self.x+self.true_prob[k], self.true_prob[k]), label='cdf')
                axes[k].title.set_text('Arm'+str(k+1)+'('+str(self.true_prob[k])+')')
                axes[k].legend(loc='upper left', prop={"size":8})
            plt.savefig("ecdf_"+str(self.sample_size)+"_"+self.algo+".png")
        else:
            pass

        return total_stats

    def total_outputs(self):
        total_stats = self.compute_total_stats()
        total_stats_ot = pd.DataFrame(total_stats)
        total_stats_ot.to_csv(self.folder + self.sys_sep + "boot_total" + "_ot" + ".csv")

