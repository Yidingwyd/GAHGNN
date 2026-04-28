# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:38:04 2024

@author: YidingWang
"""

import math
import pandas as pd
import numpy as np
from itertools import combinations





num = 100
elem_dict = {'Al':np.linspace(-np.pi/2, np.pi/2, num, endpoint=True),
             'Cr':np.linspace(-np.pi/2, np.pi/2, num, endpoint=True),
             'Mo':np.linspace(-np.pi/2, np.pi/2, num, endpoint=True)}



for name1, name2 in combinations(elem_dict.keys(), 2):
    elem_list1 = elem_dict[name1]
    elem_list2 = elem_dict[name2]
    id_list, comp_list, target_list = [], [], []
    i = 0
    
    
    for elem1 in elem_list1:
        for elem2 in elem_list2:
            id_list.append(i)
            composition = {}
            for k in elem_dict.keys():
                composition[k] = 0
            
            composition[name1] = elem1
            composition[name2] = elem2
        
            comp_list.append(composition)
            target_list.append(0)
            i += 1
    df = pd.DataFrame({'material_id':id_list,
                       'composition':comp_list,
                       'target':target_list})
    df.to_csv('./0-0/'+ name1+ name2 +'.csv', index=False)
    print("python predict.py --sample ./example/0-0/"+name1+name2+".csv --embedding ./example/onehot-embedding.json --modelpath ./example/0-0/best.pth.tar --savepath ./example/0-0/"+name1+name2+".xlsx")




                

    
    
    
    




