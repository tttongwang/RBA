# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 02:18:16 2022

@author: tongw
"""

# 1. run 1 time lin algo save X, reward
# 
# 2. for each b
# sample row of X, it follows binomial, if X bern, permutation is limited, match X get r
# save XTX, est_theta

import numpy as np
import pandas as pd
import itertools

from LinUCB                 import LinUCB
from LinUCB_boot            import LinUCB_boot
from generate_theta         import theta
from generate_common_paras  import B, feature_d, K, sample_size, linucb_alpha, Repeats


def get_all_combination_bern(feature_d):
    return np.float32(list(itertools.product([0, 1], repeat=feature_d)))



def save_contextual_results():
    while True:
        context_bandits = LinUCB(theta=theta, linucb_alpha=linucb_alpha, feature_d=feature_d, K=K, sample_size=sample_size)
        (est_theta, output_reward, ba, X) = context_bandits.run()

        dd_total = dict.fromkeys(range(K))
        condition_in_while = False
        
        for k in range(K):
            condition_in_for = False
            Xk = X[k]
            Yk = output_reward[:,k][np.logical_not(np.isnan(output_reward[:,k]))]
        
        #if condition returns False, AssertionError is raised:
            assert Xk.shape[0] == Yk.shape[0]
        
            Nk = Xk.shape[0]
            #print('k:', k)
            #print('Nk:', Nk)
            Yk_new = np.reshape(Yk,(Nk,1))
            Mk_new = np.concatenate((Xk,Yk_new), axis=1)
        
        # key is the context X 
        # value is responses
        # cc is the frequency for each X combination
            (unique_rows, cc) = np.unique(Xk, return_counts=True, axis=0)
            ddK = {}
            all_comb_rows = get_all_combination_bern(feature_d)
            if all_comb_rows.shape[0]!=unique_rows.shape[0]:
                condition_in_while = True
                break
            

            
            for row in all_comb_rows:
                ddK[str(row)]=[]

            for j in range(Nk):
                #print(str(Mk_new[j,0:feature_d]))
                ddK[str(Mk_new[j,0:feature_d])].append(Mk_new[j,feature_d])
        
            dd_total[k] = ddK
            
            
            
            if k == K-1:
                condition_in_for = True
                break
            
        if condition_in_for:
            break
        
        if condition_in_while:
            continue

        
    return est_theta, dd_total


def run_boot_exp():
    
    (est_theta, dd_total) = save_contextual_results()
    ###################################################################
    est_theta_total = dict.fromkeys(range(K))
    XTX_det_total = dict.fromkeys(range(K))
    for k in range(K):
        est_theta_total[k] = np.zeros((B,feature_d))
        XTX_det_total[k] = np.zeros(B)
        
        
    for b in range(B):
        
        boot_bandits = LinUCB_boot(theta=theta, linucb_alpha=linucb_alpha,
                                      feature_d=feature_d, K=K, sample_size=sample_size, dd_total=dd_total)
        (est_theta, output_reward, ba, X) = boot_bandits.run()
        
        for k in range(K):
            est_theta_total[k][b,:] = est_theta[k]
            XTX_det_total[k][b] = np.linalg.det(np.dot(X[k].T,X[k]))
            
    
    ######################################################################
    
    
    # initialize for bias summary
    bias_list = np.zeros((K,feature_d))
    
    for k in range(K):
        exp_XTX_det_k = np.mean(XTX_det_total[k])
        for d in range(feature_d):
            cov_kd = np.cov(est_theta_total[k][:,d],XTX_det_total[k])[0,1]
            bias_kd = cov_kd/exp_XTX_det_k
            boot_est_bias = est_theta[k][d] + bias_kd - theta[k][d]
            bias_list[k,d] = boot_est_bias
    
    return bias_list
    
def run_multi_boot_exp():
    bias_dict = dict.fromkeys((range(K)))
    bias_summary = np.zeros((K,feature_d))
    std_summary = np.zeros((K,feature_d))

    for k in range(K):
        bias_dict[k] = np.zeros((Repeats,feature_d))
        
    for r in range(Repeats):
        bias_list = run_boot_exp()
        for k in range(K):
            bias_dict[k][r,:] = bias_list[k]
            
            
    for k in range(K):
        bias_summary[k,:] = np.mean(bias_dict[k], axis=0)
        std_summary[k,:] = np.std(bias_dict[k], axis=0)
    return bias_summary, std_summary



(bias_summary, std_summary) = run_multi_boot_exp()

avg_output_ot_b = pd.DataFrame(bias_summary)
avg_output_ot_b.to_csv("LinUCB_resample_bias" + ".csv")

avg_output_ot_s = pd.DataFrame(std_summary)
avg_output_ot_s.to_csv("LinUCB_resample_bias_std" + ".csv")



















