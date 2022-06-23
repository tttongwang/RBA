import numpy as np
import pandas as pd

from LinUCB_W               import LinUCB_W
from generate_lambda_LinUCB import W_lambda_list
from generate_theta         import theta
from generate_common_paras  import feature_d, K, sample_size, linucb_alpha, Repeats



match_ind = 0
for n in range(W_lambda_list.shape[1]):
    for m in range(W_lambda_list.shape[1]):
        #print([n,m])
        fixed_lambda = [W_lambda_list[0,n],W_lambda_list[1,m]]

        # initialize for bias matrix for K arms
        bias   = dict.fromkeys(range(K))
        bias_W = dict.fromkeys(range(K))
    
        for i in range(K):
            bias[i]   = np.zeros((Repeats,feature_d))
            bias_W[i] = np.zeros((Repeats,feature_d))
        
        # initialize for bias summary
        bias_list   = np.zeros((K,feature_d))
        std_list    = np.zeros((K,feature_d))
        bias_list_W = np.zeros((K,feature_d))
        std_list_W  = np.zeros((K,feature_d))
        
        
        output_reward_dict = dict.fromkeys((range(Repeats)))
        ba_dict            = dict.fromkeys((range(Repeats)))
        X_dict             = dict.fromkeys((range(Repeats)))
        
        for r in range(Repeats):
            context_bandits = LinUCB_W(theta        = theta, 
                                       linucb_alpha = linucb_alpha, 
                                       feature_d    = feature_d, 
                                       K            = K, 
                                       sample_size  = sample_size, 
                                       fixed_lambda = fixed_lambda)
            (est_theta, output_reward, ba, X, est_theta_W) = context_bandits.run()
            
            output_reward_dict[r] = output_reward
            ba_dict[r]            = ba
            X_dict[r]             = X
            
            
            
            for key in range(K):
                bias[key][r,:]   = est_theta[key] - theta[key]
                bias_W[key][r,:] = est_theta_W[key] - theta[key]
        
        for key in range(K):
            bias_list[key,:]   = np.mean(bias[key],axis=0)
            std_list[key,:]    = np.std(bias[key],axis=0)
            bias_list_W[key,:] = np.mean(bias_W[key],axis=0)
            std_list_W[key,:]  = np.std(bias_W[key],axis=0)
        
        avg_output_ot_b = pd.DataFrame(bias_list)
        avg_output_ot_b.to_csv("LinUCB_bias" + ".csv")
        avg_output_ot_s = pd.DataFrame(std_list)
        avg_output_ot_s.to_csv("LinUCB_std" + ".csv")        
        
        
        if (np.mean(std_list_W[0,:])-np.mean(std_list[0,:]) < 0.5) and (np.mean(std_list_W[0,:])-np.mean(std_list[0,:]) >0):
            match_ind += 1
            avg_output_ot_bw = pd.DataFrame(bias_list_W)
            avg_output_ot_bw.to_csv("LinUCB_bias_" + 'lambda_' + str(fixed_lambda) +".csv")
            avg_output_ot_sw = pd.DataFrame(std_list_W)
            avg_output_ot_sw.to_csv("LinUCB_std_" + 'lambda_' + str(fixed_lambda) +".csv")
            
if match_ind == 0:
    print('no matched lambda!')






############################################################################
bias   = dict.fromkeys(range(K))
bias_W = dict.fromkeys(range(K))
    
for i in range(K):
    bias[i]   = np.zeros((Repeats,feature_d))
    bias_W[i] = np.zeros((Repeats,feature_d))
    
# initialize for bias summary
bias_list   = np.zeros((K,feature_d))
std_list    = np.zeros((K,feature_d))
bias_list_W = np.zeros((K,feature_d))
std_list_W  = np.zeros((K,feature_d))
    
    
output_reward_dict = dict.fromkeys((range(Repeats)))
ba_dict            = dict.fromkeys((range(Repeats)))
X_dict             = dict.fromkeys((range(Repeats)))
    
for r in range(Repeats):
    context_bandits = LinUCB_W(theta=theta, 
                               linucb_alpha = linucb_alpha, 
                               feature_d    = feature_d, 
                               K            = K,
                               sample_size  = sample_size,
                               fixed_lambda = False)
    (est_theta, output_reward, ba, X, est_theta_W) = context_bandits.run()
    
    output_reward_dict[r] = output_reward
    ba_dict[r]            = ba
    X_dict[r]             = X
    
    for key in range(K):
        bias[key][r,:]   = est_theta[key] - theta[key]
        bias_W[key][r,:] = est_theta_W[key] - theta[key]
    
for key in range(K):
    bias_list[key,:]   = np.mean(bias[key],axis=0)
    std_list[key,:]    = np.std(bias[key],axis=0)
    bias_list_W[key,:] = np.mean(bias_W[key],axis=0)
    std_list_W[key,:]  = np.std(bias_W[key],axis=0)


avg_output_ot_bw = pd.DataFrame(bias_list_W)
avg_output_ot_bw.to_csv("LinUCB_bias_W" +".csv")
avg_output_ot_sw = pd.DataFrame(std_list_W)
avg_output_ot_sw.to_csv("LinUCB_std_W" +".csv")



