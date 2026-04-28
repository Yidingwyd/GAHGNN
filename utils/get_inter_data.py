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
elem_dict = {'W':np.linspace(5, 11, num, endpoint=True),
             'Mo':np.linspace(0, 3, num, endpoint=True),
             'Zr':np.linspace(0, 2, num, endpoint=True)}



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
    df.to_csv('./1/0-0/'+ name1+ name2 +'.csv', index=False)
    print("python predict.py --sample ./Kfold/WJPXY_1450_new/1/0-0/"+name1+name2+".csv --embedding ./Kfold/WJPXY_1450_new/Nb_ebd.json --modelpath ./Kfold/WJPXY_1450_new/0-0best.pth.tar --savepath ./Kfold/WJPXY_1450_new/1/0-0/"+name1+name2+".xlsx")




                

    
    
    
    




