import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean
import torch.nn.init as init
from entmax import budget_bisect

class MeanPooling(nn.Module):
    """Mean pooling"""

    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return scatter_mean(x, index, dim=0)

    def __repr__(self):
        return self.__class__.__name__


class SumPooling(nn.Module):
    """Sum pooling"""

    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return scatter_add(x, index, dim=0)

    def __repr__(self):
        return self.__class__.__name__


class MaxPooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return scatter_max(x, index, dim=0)[0]

    def __repr__(self):
        return self.__class__.__name__


class TopKPooling(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, score, x, index):
        score_abs = torch.abs(score)
        unique_groups = torch.unique(index)
        num_groups = unique_groups.size(0)
        pooled = torch.zeros(num_groups, x.size(1), device=x.device)

        for i, group in enumerate(unique_groups):
           
            mask = index == group
            group_scores = score_abs[mask].view(-1,)

            group_x = x[mask]

            if group_scores.numel() >= self.k:

                topk_scores, topk_idx = group_scores.topk(self.k, largest=True, sorted=True)
                pooled[i] = group_x[topk_idx].sum(dim=0)
            else:

                pooled[i] = group_x.sum(dim=0)

        return pooled.view(-1, 1)


    def __repr__(self):
        return self.__class__.__name__

class BudgetPooling(nn.Module):

    def __init__(self, gate_nn, message_nn):
        """
        Args:
            gate_nn: Variable(nn.Module)
            message_nn
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, x_alpha, x, index, b):
        gate = self.gate_nn(x_alpha)
        gate = budget_bisect(gate, budget= b, dim = 1)
        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out, gate, x

    def __repr__(self):
        return self.__class__.__name__


class AttentionPooling(nn.Module):
    """
    softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Args:
            gate_nn: Variable(nn.Module)
            message_nn
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, x_alpha, x, index):
        gate = self.gate_nn(x_alpha)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out, gate, x

    def __repr__(self):
        return self.__class__.__name__

def group_topk_pure_pytorch_2d(src, index, k=2):

    N, C = src.shape
    assert C == 1

    src_flat = src.squeeze(1) 


    unique_groups, inverse_indices = torch.unique(index, sorted=True, return_inverse=True)

    sorted_src, sorted_indices = torch.sort(src_flat, descending=True)
    sorted_group = inverse_indices[sorted_indices] 


    group_ids, group_lengths = torch.unique(sorted_group, return_counts=True)

    group_start = torch.cumsum(group_lengths, dim=0) - group_lengths  


    arange = torch.arange(N, device=src.device)


    ranks = arange - torch.repeat_interleave(group_start, group_lengths)

 
    mask = ranks < k  


    topk_sorted_indices = sorted_indices[mask]


    final_mask = torch.zeros_like(src_flat, dtype=torch.bool)
    final_mask[topk_sorted_indices] = True


    final_mask = final_mask.unsqueeze(1)  


    result = src.clone()
    result[~final_mask] = 0

    return result


def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min)/(f_max - f_min + eps) + index.float()*(-1)**int(descending)
    perm = norm.argsort(dim=dim, descending=descending)

    return src[perm], perm

class WeightedAttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, index, weights):
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]

        gate = weights * self.pow * gate.exp()
        assert torch.isfinite(gate).all()

        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)

        out = scatter_add(gate * x, index, dim=0)

        return out, gate, x

    def __repr__(self):
        return self.__class__.__name__


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.LeakyReLU,
        batchnorm=False,
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims
        # print(dims)
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)
        
        self._initialize_weights()
        
    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)
    
    def _initialize_weights(self):
        for fc in self.fcs:
            init.kaiming_normal_(fc.weight, nonlinearity='leaky_relu')
            if fc.bias is not None:
                init.constant_(fc.bias, 0)
    
    def __repr__(self):
        return self.__class__.__name__

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()

        self.fc_out.reset_parameters()


class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.ReLU,
        batchnorm=False,
        return_features=False,
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.return_features = return_features
        if not self.return_features:
            self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        if self.return_features:
            return x
        else:
            return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__
