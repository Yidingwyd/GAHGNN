# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:28:42 2024

@author: YidingWang
"""

import json
import pandas as pd
from pymatgen.core import Composition

dataset = pd.read_excel('origin_dataset.xlsx')
columns = dataset.columns[0:-1]

comp_list, target_list = [], []
for i in dataset.index:
    comp_dict = {}
    for c in columns:
        comp_dict[c] = dataset.loc[i, c]
    comp_list.append(comp_dict)
    target_list.append(dataset.loc[i, 'target'])

ids = [i for i in range(len(comp_list))]
data = {'material_id':ids,
        'composition':comp_list,
        'target':target_list}

HANS_dataset = pd.DataFrame(data)

HANS_dataset.to_csv('./my_dataset.csv', index= False)



