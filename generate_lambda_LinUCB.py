# -*- coding: utf-8 -*-
"""
@author: tongw

Assign lambda candidates for different arms to choose the qualified lambda:
    0 for arm0 
    1 for arm1
    extend W_lambda_list if there is more arms

 
"""

import numpy as np

W_lambda_list0 = [4,5,6,7,8,9,10,11,12]
W_lambda_list1 = [4,5,6,7,8,9,10,11,12]
W_lambda_list  = np.array([W_lambda_list0, W_lambda_list1])




