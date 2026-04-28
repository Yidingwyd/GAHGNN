# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:38:04 2024

@author: YidingWang
"""

import pandas as pd
import numpy as np


# For EAF dataset
num = 100
elem_dict = {'Al':np.linspace(-np.pi/2, np.pi/2, num, endpoint=True),
             'Cr':np.linspace(-np.pi/2, np.pi/2, num, endpoint=True),
             'Mo':np.linspace(-np.pi/2, np.pi/2, num, endpoint=True)}

# For HEAC dataset

# elem_dict = {'one':[26,27,28,29,30],
#              'two':[26,27,28,29,30],
#              'three':[26,27,28,29,30],
#              'four':[26,27,28,29,30],
#              'five':[26,27,28,29,30],
#              'six':[26,27,28,29,30],
#              'seven':[26,27,28,29,30],
#              'eight':[26,27,28,29,30],
#              'nine':[26,27,28,29,30],
#              'ten':[26,27,28,29,30],
#              'eleven':[26,27,28,29,30],
#              'twelve':[26,27,28,29,30],
#              'thirteen':[26,27,28,29,30],
#              'fourteen':[26,27,28,29,30],
#              'fifteen':[26,27,28,29,30],
#              'sixteen':[26,27,28,29,30]}

# For Nb dataset

# num = 100
# elem_dict = {'W':np.linspace(5, 11, num, endpoint=True),
#              'Mo':np.linspace(0, 3, num, endpoint=True),
#              'Zr':np.linspace(0.05, 2, num, endpoint=True),
#              'C':np.linspace(0, 0.2, num, endpoint=True),
#              'Hf':np.linspace(0, 7, num, endpoint=True),
#              'Ta':np.linspace(0, 10, num, endpoint=True),
#              'N':np.linspace(0, 0.3, num, endpoint=True)}

comp_list, target_list = [], []

for i in range(num):    
        comp_dict = {}
        for c in elem_dict.keys():
            comp_dict[c] = elem_dict[c][i]
        comp_list.append(comp_dict)
        target_list.append(0)

    
    
ids = [i for i in range(len(comp_list))]
data = {'material_id':ids,
        'composition':comp_list,
        'target':target_list}

HANS_dataset = pd.DataFrame(data)

HANS_dataset.to_csv('./main.csv', index= False)









                

    
    
    
    




