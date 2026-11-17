import torch
from torch import nn
from entmax import entmax_bisect
from torch_scatter import scatter_mean
from model_base import HyperLayer
ARCHITECTURE = dict(elem_fea_len=64, edge_f=[64], edge_h=[64, 64], readout_hidden=[64, 64], node_batchnorm=False, attention_init='small')

class EntmaxHyperLayer(HyperLayer):

    def __init__(self, feature_dim, alpha_init=1.5, learn_alpha=False, score_skip=False):
        super().__init__(comp_fea_len=feature_dim, edge_heads=2, orth_loss=False, **ARCHITECTURE)
        nn.init.constant_(self.alpha, alpha_init)
        self.alpha.requires_grad_(learn_alpha)
        nn.init.zeros_(self.constant)
        self.constant.requires_grad_(False)
        nn.init.zeros_(self.ent_gate.message_nn.fc_out.weight)
        nn.init.zeros_(self.ent_gate.message_nn.fc_out.bias)
        self.mix = 1.0
        self.floor = 0.0
        self.temperature = 1.0
        self.excluded_supports = ()
        self.selector_skip = nn.Linear(feature_dim, 1, bias=False) if score_skip else None
        if self.selector_skip is not None:
            nn.init.zeros_(self.selector_skip.weight)

    def configure_exclusions(self, supports):
        self.excluded_supports = tuple(tuple(sorted(support)) for support in supports)

    def forward(self, weights, features, comp_idx, graph_idx, material_idx):
        nodes = torch.cat([self.embedding(features), weights], dim=1)
        pooled = torch.stack([pool(nodes, nodes, graph_idx)[0] for pool in self.edge_pool]).mean(0)
        raw = self.embedding_edge(pooled)
        logits = self.ent_gate(features, features, graph_idx)[0]
        if self.selector_skip is not None:
            logits = logits + self.selector_skip(scatter_mean(features, graph_idx, dim=0))
        n_edges = int((material_idx == 0).sum().item())
        n_samples = int(material_idx.max().item()) + 1
        logits = logits.view(n_samples, n_edges, 1)
        alpha = self.alpha.clamp(1.1, 1.95)
        if self.excluded_supports:
            members = comp_idx.reshape(n_samples, n_edges, -1).sort(dim=2).values
            allowed = torch.ones((n_samples, n_edges), dtype=torch.bool, device=logits.device)
            for support in self.excluded_supports:
                forbidden = torch.as_tensor(support, dtype=members.dtype, device=members.device)
                if forbidden.numel() != members.shape[2]:
                    raise ValueError("Excluded effect order does not match this hypergraph layer.")
                allowed = allowed & (members != forbidden).any(dim=2)
            uniform = (allowed.to(logits.dtype) / allowed.sum(1, keepdim=True).clamp_min(1)).unsqueeze(2)
            if n_edges > 1:
                # A dummy candidate keeps entmax finite for an entirely forbidden row.
                safe_allowed = allowed.clone()
                safe_allowed[:, 0] |= ~allowed.any(dim=1)
                masked_logits = (logits / self.temperature).masked_fill(~safe_allowed.unsqueeze(2), -float("inf"))
                p = entmax_bisect(masked_logits, alpha, dim=1).masked_fill(~allowed.unsqueeze(2), 0.0)
            else:
                p = allowed.to(logits.dtype).unsqueeze(2)
        else:
            p = entmax_bisect(logits / self.temperature, alpha, dim=1) if n_edges > 1 else torch.ones_like(logits)
            uniform = torch.ones_like(p) / n_edges
            allowed = torch.ones((n_samples, n_edges), dtype=torch.bool, device=logits.device)
        gate = (1 - self.mix) * uniform + self.mix * ((1 - self.floor) * p + self.floor * uniform)
        raw = raw.view(n_samples, n_edges, 1)
        terms = gate * raw
        self.last_gate = gate[0, :, 0]
        self.last_entmax = p[0, :, 0]
        self.last_logits = logits[0, :, 0]
        self.last_allowed = allowed[0]
        self.last_raw = raw[:, :, 0]
        return (terms.sum(1), terms[:, :, 0], raw[:, :, 0])
