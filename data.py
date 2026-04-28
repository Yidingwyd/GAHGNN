# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 19:47:02 2024

@author: YidingWang
"""

import json,  torch, tqdm
from torch.utils.data import Dataset
import pandas as pd
from itertools import combinations

class HYPERDataset(Dataset):
    def __init__(self, data_path, embedding_file, max_graph_size = 3, exclusion = None):
        with open(embedding_file) as f:
            self.embedding = json.load(f)
            f.close()
        self.df = pd.read_csv(data_path)
        self.max_graph_size = max_graph_size
        self.exclusion = exclusion
        self.final_dataset = []
        print('Loading Dataset...')
        for i in tqdm.tqdm(self.df.index, ncols=50):
            self.final_dataset = self.final_dataset + self.get_graphs(self.df.loc[i, 'composition'],
                                                      self.df.loc[i, 'target'])
        print('Done.')
                    
        
    def get_graphs(self, composition, target):
        composition = eval(composition)
        if len(composition) < self.max_graph_size:
            print(composition)
            print('请整理数据或更改最大子图尺寸')
            return []
        else:
            comp_graphs = self.get_size_graphs(composition, self.max_graph_size)
            return [(comp_graphs, torch.Tensor([target]), composition)]
    
    def get_size_graphs(self, comp, size):
        size_graphs = []
        keys = list(comp.keys())
        for i, elements in enumerate(combinations(comp.keys(), size)):
            
            if self.exclusion is not None:
                if set(elements) in self.exclusion:
                    continue
            comp_weights = []
            comp_idx = []
            for e in elements:
                comp_weights.append([comp[e]])
                comp_idx.append(keys.index(e))
            comp_fea = self.get_single_graph(elements)
            graph = (torch.Tensor(comp_weights),
                     torch.Tensor(comp_fea),
                     torch.Tensor(comp_idx))
            size_graphs.append(graph)
        return size_graphs
                                
    def get_single_graph(self, elements):
        comp_fea = []
        for e in elements:
            comp_fea.append(self.embedding[e])     
        return comp_fea
    def __len__(self):
        return len(self.final_dataset)
    def __getitem__(self, i):
        return self.final_dataset[i]

def collate_batch_origin(dataset_list):
    comp_graphs, _, _ = dataset_list[0]
    size = len(comp_graphs[0][0])
    
  
    composition_list = [] 
    batch_target = []
    

    batch_comp_weights = []
    batch_comp_fea = []
    batch_comp_idx = []
    graph_idx = []

    material_idx = []
    base_graph_idx = 0
    max_idx = 0
    for j, (size_graphs, target, composition) in enumerate(dataset_list):

        for k, graph in enumerate(size_graphs):
            comp_weights,comp_fea,comp_idx = graph
            batch_comp_weights.append(comp_weights)
            batch_comp_idx.append((comp_idx + max_idx).long())
            batch_comp_fea.append(comp_fea)
            graph_idx.append(torch.LongTensor([base_graph_idx]*size))
            base_graph_idx += 1
            # graph_weights.append(comp_weights.mean())
            material_idx.append(torch.LongTensor([j]))
        max_idx += max(comp_idx)+1
            

    for j, (comp_graphs, target, composition) in enumerate(dataset_list):
        composition_list.append(composition)
        batch_target.append(target)
    
    return (
                torch.cat(batch_comp_weights, dim = 0),
                torch.cat(batch_comp_fea, dim = 0),
                batch_comp_idx,
                torch.cat(graph_idx, dim = 0),
                torch.cat(material_idx, dim = 0),    
                composition_list,
                torch.cat(batch_target, dim=0).view(-1,1)
            )




