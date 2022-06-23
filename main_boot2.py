import numpy as np
from boot_experiments2 import Experiments
from settings import dist,sys_sep, true_prob, sigma, sample_size,repeats,gap,algo_list,random_choice,boot_number,infer_bool, plot_total_bool, plot_bool, x, lin, lin_t, lin_const, dp, dp_eps, gd_eps, lil_edlta, lil, beta, lil_eps, lil_lambda



for algo in algo_list:
    experiments = Experiments(x = x, infer_bool=infer_bool, plot_total_bool=plot_total_bool, plot_bool=plot_bool,
     boot_number=boot_number, dist = dist, sys_sep=sys_sep, algo=algo,
                lin=lin, lin_t=lin_t, lin_const=lin_const, random_choice=random_choice,
                true_prob=true_prob, sample_size=sample_size, repeats=repeats, gap=gap, 
                gd_eps=gd_eps, sigma=sigma, lil_delta=lil_delta, lil_eps=lil_eps, 
                lil_lambda=lil_lambda, lil_beta=lil_beta, dp=dp, dp_eps=dp_eps)
    experiments.set_path()
    experiments.total_outputs()




