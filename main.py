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
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from data import HYPERDataset, collate_batch_origin
from model import HyperLayer
import csv

def parse_list(input_str):
    return eval(input_str)

parser = argparse.ArgumentParser(description='Hypergraph Attention Neural Network for Stoichiometry')
parser.add_argument('--elem_fea_len', default=256, type=int)
parser.add_argument('--edge_f', default=[256], type=parse_list)#边注意力层中f的中间层结构
parser.add_argument('--edge_h', default=[256, 128], type=parse_list)#边注意力层中h的中间层结构
parser.add_argument('--gate', default=[256,128], type=parse_list)#gate层的结构
parser.add_argument('--size', default=3, type=int)
parser.add_argument('--head', default=2, type=int)
parser.add_argument('--k', default=0, type=int)
parser.add_argument('--exclusion', default=None, type=parse_list)
parser.add_argument('--orth_loss', action='store_true')

parser.add_argument('--train_data')
parser.add_argument('--val_data')
parser.add_argument('--embedding')
parser.add_argument('--savepath')
parser.add_argument('--gamma', default=0.7)
parser.add_argument('--l1', default=1e-3, type=float )
parser.add_argument('--l2', default=1e0, type=float )
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 4000)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate1', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                        '3e-4)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')


args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()





def main():
    global args
    print(os.getcwd())
    
    train_dataset = HYPERDataset(args.train_data, args.embedding, args.size, args.exclusion)
    val_dataset = HYPERDataset(args.val_data, args.embedding, args.size, args.exclusion)
    collate_batch = collate_batch_origin
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size,
                              collate_fn = collate_batch, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size,
                              collate_fn = collate_batch, shuffle = False)
    

    sample_data_list = [train_dataset[i] for i in range(len(train_dataset))]
    sample_target = collate_batch_origin(sample_data_list)[6]
    
   
    if args.cuda:
        sample_target = sample_target.to(torch.device('cuda:0'))
    normalizer = Normalizer(sample_target)

 

    comp_fea_len = train_dataset[0][0][0][1].shape[-1]

    model = HyperLayer(comp_fea_len = comp_fea_len,
                        elem_fea_len = args.elem_fea_len,
                        edge_f = args.edge_f,
                        edge_h = args.edge_h,
                        gate = args.gate,
                        k = args.k,
                        edge_heads = args.head,
                        orth_loss = args.orth_loss) 
    if args.cuda:
        model.cuda()

   
    criterion = nn.HuberLoss(delta=1.0)
        
    optimizer = optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

   
    scheduler = ReduceLROnPlateau(
                                    optimizer,
                                    mode='min',       
                                    factor=0.5,        
                                    patience=10,       
                                    threshold=1e-3,   
                                    min_lr=1e-5
                                )
    
    best_loss = 1e10
    with open(args.savepath + 'out.csv', mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_mse', 'val_loss', 'val_mse', 'effect', 'alpha'])
    for epoch in range(args.epochs):
        train_loss, train_mse = train(train_loader, model, criterion, optimizer, epoch, normalizer)
        val_loss, val_mse, effect, alpha = validate(val_loader, model, criterion, epoch, normalizer)
        new_data = [epoch, train_loss, train_mse, val_loss, val_mse, effect, alpha]
        with open(args.savepath + 'out.csv', mode="a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(new_data)
        
        scheduler.step(train_mse)
        if val_loss < best_loss:
            torch.save({'epoch':epoch,
                       'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'normalizer': normalizer.state_dict(),
                       'args': vars(args)}
                       , args.savepath + 'best.pth.tar')
            best_loss = val_loss
    
    
        

def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mse_errors = AverageMeter() 


    model.train()

    end = time.time()
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        device = torch.device('cuda:0') if args.cuda else torch.device('cpu')

        comp_weights, comp_fea, comp_idx, graph_idx, \
            material_idx, composition_list, target = batch_data
        comp_weights = comp_weights.to(device)
        comp_fea = comp_fea.to(device)
        comp_idx_list = [d.to(device) for d in comp_idx]
        comp_idx = comp_idx_list
        graph_idx = graph_idx.to(device)
        material_idx = material_idx.to(device)
        target = target.to(device)
        input_var = (comp_weights, comp_fea, comp_idx, graph_idx, material_idx)
   

        target_normed = normalizer.norm(target)

        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        output,o_term,_, contribution_list, alpha = model(*input_var)
        loss = criterion(output, target_var) + args.l1*epoch/args.epochs * o_term + args.l2 *(alpha-2)**2
        
        mse_error = mse(normalizer.denorm(output.data), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mse_errors.update(mse_error.cpu().item(), target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'MSE {mse_errors.val:.3f} ({mse_errors.avg:.3f})'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, mse_errors=mse_errors)
        )
        

    return losses.avg, mse_errors.avg


def validate(val_loader, model, criterion, epoch, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mse_errors = AverageMeter()
    
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, batch_data in enumerate(val_loader):

        with torch.no_grad():
            device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
             
            comp_weights, comp_fea, comp_idx, graph_idx, \
                material_idx, composition_list, target = batch_data
            comp_weights = comp_weights.to(device)
            comp_fea = comp_fea.to(device)
            comp_idx_list = [d.to(device) for d in comp_idx]
            comp_idx = comp_idx_list
            graph_idx = graph_idx.to(device)
            material_idx = material_idx.to(device)
            target = target.to(device)
            input_var = (comp_weights, comp_fea, comp_idx, graph_idx, material_idx)
                    
        target_normed = normalizer.norm(target)

        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)
        output,o_term,effect, _, alpha = model(*input_var)
        loss = criterion(output, target_var) + args.l1 *epoch/args.epochs * o_term + args.l2 *(alpha-2)**2
        mse_error = mse(normalizer.denorm(output.data), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mse_errors.update(mse_error.cpu().item(), target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data)
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
        batch_time.update(time.time() - end)
        end = time.time()


        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'MSE {mse_errors.val:.3f} ({mse_errors.avg:.3f})'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses,
            mse_errors=mse_errors))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    
    
    print(' {star} MSE {mse_errors.avg:.3f}'.format(star=star_label,
                                                    mse_errors=mse_errors))
    return losses.avg, mse_errors.avg, effect,float(alpha.cpu().detach())

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


def mse(prediction, target):
    return F.mse_loss(prediction, target)


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def column_wise_mae(output: torch.Tensor, target: torch.Tensor) -> list:   
    absolute_errors = torch.abs(output - target)
    mae_per_column = torch.mean(absolute_errors, dim=0)
    return mae_per_column.cpu().tolist()


if __name__ == '__main__':
    main()
