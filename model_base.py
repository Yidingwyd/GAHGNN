import torch.nn as nn
import torch
from roost_segments import SimpleNetwork, AttentionPooling
from torch_scatter import scatter_add
from entmax import entmax_bisect

def orthogonal_term(X):
    X = X.squeeze(-1)
    G = X.T @ X
    idx = torch.triu_indices(X.size()[1], X.size()[1], offset=1)
    out = G[idx[0], idx[1]].view(-1, 1)
    return out

class HyperLayer(nn.Module):

    def __init__(self, comp_fea_len, elem_fea_len=256, edge_f=[256], edge_h=[256, 256], gate=[256, 128], k=0, edge_heads=5, orth_loss=False, readout_hidden=None, node_batchnorm=True, attention_init='uniform', attention_init_scale=0.02):
        super(HyperLayer, self).__init__()
        self.embedding = nn.Linear(comp_fea_len, elem_fea_len - 1)
        self.orth_loss = orth_loss
        self.edge_pool = nn.ModuleList([AttentionPooling(gate_nn=SimpleNetwork(elem_fea_len, 1, edge_f, activation=nn.Tanh, batchnorm=node_batchnorm), message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, edge_h, activation=nn.Tanh, batchnorm=node_batchnorm)) for _ in range(edge_heads)])
        readout_hidden = list(readout_hidden or [])
        if readout_hidden:
            dims = [elem_fea_len] + readout_hidden
            layers = []
            for (n_in, n_out) in zip(dims[:-1], dims[1:]):
                layers.extend([nn.Linear(n_in, n_out), nn.SiLU()])
            layers.append(nn.Linear(dims[-1], 1))
            self.embedding_edge = nn.Sequential(*layers)
        else:
            self.embedding_edge = nn.Linear(elem_fea_len, 1)
        self.ent_gate = AttentionPooling(gate_nn=SimpleNetwork(comp_fea_len, 1, edge_f, activation=nn.Sigmoid, batchnorm=node_batchnorm), message_nn=SimpleNetwork(comp_fea_len, 1, edge_h, activation=nn.Sigmoid, batchnorm=node_batchnorm))
        self.alpha = nn.Parameter(torch.Tensor([1.8]))
        self.constant = nn.Parameter(torch.randn(1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if attention_init not in {'uniform', 'small', 'xavier'}:
            raise ValueError('attention_init must be uniform, small, or xavier')
        for pool in self.edge_pool:
            if attention_init == 'uniform':
                nn.init.zeros_(pool.gate_nn.fc_out.weight)
            elif attention_init == 'small':
                nn.init.normal_(pool.gate_nn.fc_out.weight, std=attention_init_scale)
            nn.init.zeros_(pool.gate_nn.fc_out.bias)

    def forward(self, comp_weights, comp_fea, comp_idx, graph_idx, material_idx):
        elem_fea = self.embedding(comp_fea)
        elem_fea = torch.cat([elem_fea, comp_weights], dim=1)
        edge_head_fea = []
        for edge_head in self.edge_pool:
            (e, _, _) = edge_head(elem_fea, elem_fea, graph_idx)
            edge_head_fea.append(e)
        edge_fea = torch.mean(torch.stack(edge_head_fea), dim=0)
        edge_fea = self.embedding_edge(edge_fea)
        (edge_z, _, _) = self.ent_gate(comp_fea, comp_fea, graph_idx)
        edge_num = int((material_idx == 0).sum().item())
        n_materials = int(material_idx.max().item()) + 1
        edge_z = edge_z.view(n_materials, edge_num, -1)
        alpha = torch.clamp(self.alpha, min=1.0, max=2.0)
        gate = entmax_bisect(edge_z, alpha, dim=1)
        effect = gate[0].squeeze().tolist()
        edge_fea = edge_fea.view(gate.size())
        contribution = torch.mul(gate, edge_fea)
        if self.orth_loss:
            o_term = orthogonal_term(contribution).abs().mean()
        else:
            o_term = 0
        out = scatter_add(contribution.view(-1, 1), material_idx, dim=0)
        out = out + self.constant
        return (out, o_term, effect, contribution, alpha)
