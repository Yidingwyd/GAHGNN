# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:43:37 2024

@author: YidingWang
"""

import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split
from data import HYPERDataset, collate_batch_origin
# from cgcnn_data import  get_train_val_test_loader
from model import HyperLayer
import json
import pandas as pd
from pymatgen.core import Composition
from itertools import combinations

def parse_list(input_str):
    # return list(map(int, input_str.strip('[]').split(',')))
    return eval(input_str)

parser = argparse.ArgumentParser(description='Hypergraph Attention Neural Network for Stoichiometry')

# parser.add_argument('--train_data')
# parser.add_argument('--val_data')
parser.add_argument('--sample')
parser.add_argument('--embedding')
# parser.add_argument('--savepath')
parser.add_argument('--modelpath')
parser.add_argument('--savepath')
# parser.add_argument('--seed', default=581, type=int)
# parser.add_argument('--gamma', default=0.7)
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
# parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
#                     help='number of data loading workers (default: 0)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--elem_fea_len', default=256, type=int)
# parser.add_argument('--edge_f', default=[256], type=parse_list)#边注意力层中f的中间层结构
# parser.add_argument('--edge_h', default=[256, 128], type=parse_list)#边注意力层中h的中间层结构
# parser.add_argument('--gate', default=[256,128], type=int)#gate层的结构
# parser.add_argument('--size', default=3, type=int)
# parser.add_argument('--exclusion', default=None, type=parse_list)
# parser.add_argument('--budget_anneal', action='store_true')
# parser.add_argument('--k', default=0, type=int)
# # parser.add_argument('--k_list', default=[0,0,0], type=parse_list)
# # parser.add_argument('--HEA', action='store_true')
# # parser.add_argument('--beta_list', default=[0, 0, 0], type=parse_list)



args = parser.parse_args(sys.argv[1:])
# args.cuda = not args.disable_cuda and torch.cuda.is_available()


def main():
    global args, best_mae_error
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
    else:
        print("=> no model found at '{}'".format(args.modelpath))
    saved_args_dict = checkpoint['args']

    args_network = argparse.Namespace(**saved_args_dict)
        
    dataset = HYPERDataset(args.sample, args.embedding, args_network.size, args_network.exclusion)
    
    collate_batch = collate_batch_origin
    
    test_loader = DataLoader(dataset, batch_size = args_network.batch_size,
                              collate_fn = collate_batch, shuffle = False)
    
    comp_fea_len = dataset[0][0][0][1].shape[-1]
    
    
        
    
    
    model = HyperLayer(comp_fea_len = comp_fea_len,
                        elem_fea_len = args_network.elem_fea_len,#comp_fea经过embedding的长度
                        edge_f = args_network.edge_f,#边注意力层中f的中间层结构
                        edge_h = args_network.edge_h,
                        gate = args_network.gate,#gate层的结构
                        k = args_network.k,
                        edge_heads = args_network.head,
                        orth_loss = args_network.orth_loss)
    if args_network.cuda:
        model.cuda()  
    device = torch.device('cuda:0') if args_network.cuda else torch.device('cpu')
    normalizer = Normalizer(torch.zeros(3))
    model.load_state_dict(checkpoint['state_dict'])
    normalizer.load_state_dict(checkpoint['normalizer'], device)
    
    
    
    
        

    
        
    formula_list, output_list, target_list, contribution_list = [],[],[],[]
    model.eval()
    for i, batch_data in enumerate(test_loader):
        # print(i)
        # if args.cuda:
        with torch.no_grad():
            device = torch.device('cuda:0') if args_network.cuda else torch.device('cpu')
            comp_weights, comp_fea, comp_idx, graph_idx, \
                material_idx, composition_list, target = batch_data
            comp_weights = comp_weights.to(device)
            comp_fea = comp_fea.to(device)
            comp_idx_list = [d.to(device) for d in comp_idx]
            comp_idx = comp_idx_list
            graph_idx = graph_idx.to(device)
            # graph_weights = [d.to(device) for d in graph_weights]
            material_idx = material_idx.to(device)
            input_var = (comp_weights, comp_fea, comp_idx, graph_idx, material_idx)
        
            output,_,_, contribution, _ = model(*input_var)
            output = normalizer.denorm(output.data)
            contribution = normalizer.denorm(contribution.data)
            # print(output)
            output = output.squeeze().tolist() 
            output = output if type(output) is list else [output]
            target = target.squeeze().tolist()
            target = target if type(target) is list else [target]
            formula_list = formula_list + composition_list
            output_list = output_list + output
            target_list = target_list + target
            contribution_list = contribution_list + contribution.squeeze(-1).tolist()
    
    

    df = pd.DataFrame({'composition':formula_list,
                       'output':output_list,
                       'target':target_list})
    df.to_excel(args.savepath)
    
    contribution_df = pd.DataFrame(contribution_list)
    with pd.ExcelWriter(args.savepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    # 写入第二个 sheet（Excel 的 sheet 名可自取，例如 "Sheet2"）
        contribution_df.to_excel(writer, sheet_name="Sheet2", index=False, header=False)
    
    




class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor,0,True).to(tensor.device)
        self.std = torch.std(tensor,0,True).to(tensor.device)

    def norm(self, tensor):
        # print(tensor.device)
        # print(self.mean.device)
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict, device):
        self.mean = state_dict['mean'].to(device)
        self.std = state_dict['std'].to(device)

if __name__ == '__main__':

    main()