###### parameter settings for the family of main functions ###### 
# context free scenarios:
# run python main_boot2.py
# change key paras: sample_size, boot_number, repeats, true_prob, sigma, algo_list, plot_total_bool

################################################################
import pandas as pd
from sys import platform



# set platform

if platform == "linux" or platform == "linux2":
	sys_sep = "/"
	# linux
elif platform == "darwin":
	sys_sep = "/"
	# OS X
elif platform == "win32":
	sys_sep = "\\"

################################################################
# common experiments para

sample_size = 100

boot_number = 2000

# try fewer repeats to double check code
repeats = 100


dist = "norm"

#true_prob = [2,1]
true_prob = [2,1.5,1,0.75,0.5]
scale_number = 0.5
s = pd.Series(true_prob)
true_prob = (s * scale_number).tolist()
sigma = 1



################################################################
# choose mab algorithms in context free
#algo_list = ["ts"]
algo_list = ["greedy","lil_ucb","ts","t_greedy"]




# make statistical inference test if K = 2
if len(true_prob)==2:
	infer_bool = True 
else:
	infer_bool = False




# make ecdf plot

# plot_total_bool = True: 
# ecdf plot will be generated, otherwise not
plot_total_bool = True
plot_bool = False
c= False
# x defines the variable range when generating ecdf plot
x = np.arange(-3, 3 + 0.25, 0.25)
gap = 10



################################################################
# paras used in rMAB, just ignore when applying RBA

random_choice = "uniform"

lin = False
# lin_t=T means: 
# linear(lambda(t)=1/t), otherwise linear_const (lambda(t)=K/t) will be used in rMAB
lin_t = False
lin_const = len(true_prob)


dp = False 
dp_eps = 150

gd_eps = 0.1
lil_delta = 0.005
lil_beta = 1
lil_eps = 0.01
lil_lambda = 9