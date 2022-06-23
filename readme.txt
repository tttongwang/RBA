############################################################
## main functions:

# MAB = LinUCB
# RBA: LinUCB_contextual_boot2.py 
# W-estimators: LinUCB_W_exp.py

# MAB = LinGreedy	
# RBA: LinGreedy_contextual_boot2.py
# W-estimators: LinGreedy_W_exp.py


############################################################
## key paras:

# generate_common_paras.py
# sample_size  = 500
# B            = 2000 :bootstrap number
# feature_d    = 2    :the number of features for each arm
# K                   :the number of arms
# Repeats             :the number of repeated experiments

# generate_theta.py
# theta[0], theta[1],..., theta[K-1]: set the coefficients in each arm

## paras used in W-estimators:
# Assign lambda candidates for different arms to choose the qualified lambda:

# MAB = LinUCB
# generate_lambda_LinUCB.py

# MAB = LinGreedy
# generate_lambda_LinGreedy.py
