# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:38:04 2024

@author: YidingWang
"""

import pandas as pd
import numpy as np



num = 100
elem_dict = {'W':np.linspace(5, 11, num, endpoint=True),
             'Mo':np.linspace(0, 3, num, endpoint=True),
             'Zr':np.linspace(0, 2, num, endpoint=True)}


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









                

    
    
    
    




