
import itertools
import numpy as np
import torch
from torch import nn
from common import graph_batch, quadrature_axes, normalize_excluded_effects
from model import EntmaxHyperLayer


VARIANTS = ("full", "no_purity", "no_sparse_loss", "mse_only",
            "no_dense", "no_sparse_stage")


def order_config(order, variant="full"):
    if variant not in VARIANTS or order not in (1, 2, 3):
        raise ValueError("Unknown ablation or unsupported interaction order.")
    return dict(alpha=1.5, learn_alpha=False, temp=0.4,
                sparse=[0.002, 0.01, 0.0][order - 1],
                score_lr=[0.003, 0.01, 0.003][order - 1],
                disable_purity=variant in ("no_purity", "mse_only"),
                disable_sparse_loss=variant in ("no_sparse_loss", "mse_only"),
                purity_support_batch_size=8)


class Stage(nn.Module):
    def __init__(self, cfg, feature_dim):
        super().__init__()
        self.layer = EntmaxHyperLayer(feature_dim, cfg["alpha"], cfg["learn_alpha"], score_skip=True)
        self.intercept = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        prediction, terms, raw = self.layer(*batch)
        return prediction + self.intercept, terms, raw


class JointModel(nn.Module):
    def __init__(self, feature_dim, highest_order, configs=None, excluded_effects=None, variable_names=None):
        super().__init__()
        if highest_order not in (1, 2, 3):
            raise ValueError("highest_order must be 1, 2, or 3.")
        self.highest_order = highest_order
        self.configs = configs or [order_config(k) for k in range(1, highest_order + 1)]
        if len(self.configs) != highest_order:
            raise ValueError("One configuration is required per trained order.")
        self.stages = nn.ModuleList([Stage(cfg, feature_dim) for cfg in self.configs])
        if excluded_effects and variable_names is None:
            raise ValueError("Variable names are required to configure excluded effects.")
        self.excluded_effects = normalize_excluded_effects(excluded_effects, variable_names or [], highest_order)
        if self.excluded_effects:
            indices = {name: index for index, name in enumerate(variable_names)}
            for order, stage in enumerate(self.stages, 1):
                stage.layer.configure_exclusions([tuple(indices[name] for name in group)
                                                  for group in self.excluded_effects if len(group) == order])

    def forward(self, batches):
        if len(batches) != self.highest_order:
            raise ValueError("Pass one graph batch for each trained order.")
        outputs, snapshots = [], []
        for stage, batch in zip(self.stages, batches):
            outputs.append(stage(batch))
            snapshots.append(dict(gate=stage.layer.last_gate.clone(),
                                  entmax=stage.layer.last_entmax.clone(),
                                  allowed=stage.layer.last_allowed.clone()))
        return (sum(value[0] for value in outputs), torch.cat([value[1] for value in outputs], 1),
                [value[2] for value in outputs], snapshots)


def selected_graph_batch(x, supports, device, feature_matrix):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    supports = torch.as_tensor(supports, dtype=torch.long, device=device)
    features = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    if x.ndim != 2 or supports.ndim != 2 or len(x) != len(supports):
        raise ValueError("Selected support rows must align with the input points.")
    count, order = supports.shape
    return (x.gather(1, supports).reshape(-1, 1),
            features[supports].reshape(-1, features.shape[1]), supports.reshape(-1),
            torch.arange(count, device=device).repeat_interleave(order),
            torch.arange(count, device=device))


class OrderConstraint:
    def __init__(self, order, device, cfg, bounds, variable_names, feature_matrix):
        self.order, self.device = order, device
        self.bounds = np.asarray(bounds, dtype=float)
        self.variable_names = list(variable_names)
        self.feature_matrix = np.asarray(feature_matrix, dtype=float)
        if (self.bounds.shape != (len(variable_names), 2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 1] < self.bounds[:, 0]) or order > len(variable_names)):
            raise ValueError("Invalid per-variable integration bounds or interaction order.")
        self.n = [16, 12, 8][order - 1]
        axes, weights = quadrature_axes(self.n, self.bounds)
        self.w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.batches = []
        if order == 1:
            self.main_batch = graph_batch(axes, order, device, variable_names, feature_matrix)
        else:
            supports = list(itertools.combinations(range(len(variable_names)), order))
            group_size = int(cfg.get("purity_support_batch_size", 8))
            if group_size < 1:
                raise ValueError("purity_support_batch_size must be positive.")
            midpoint = self.bounds.mean(1)
            for start in range(0, len(supports), group_size):
                selected = supports[start:start + group_size]
                points = []
                for support in selected:
                    meshes = np.meshgrid(*(axes[:, i] for i in support), indexing="ij")
                    x = np.repeat(midpoint[None, :], self.n ** order, axis=0)
                    for index, mesh in zip(support, meshes):
                        x[:, index] = mesh.ravel()
                    points.append(x)
                rows = np.repeat(np.asarray(selected), self.n ** order, axis=0)
                self.batches.append((start, len(selected), selected_graph_batch(
                    np.concatenate(points), rows, device, feature_matrix)))

    def __call__(self, stage, full_gates):
        n, w = self.n, self.w
        if self.order == 1:
            terms = stage(self.main_batch)[1]
            return ((terms * w[:, None]).sum(0).square()).sum()
        result = stage.intercept.new_zeros(())
        for start, count, batch in self.batches:
            raw = stage(batch)[2][:, 0]
            for local in range(count):
                term = (raw[local * n ** self.order:(local + 1) * n ** self.order]
                        * full_gates[start + local]).reshape(*([n] * self.order))
                for axis in range(self.order):
                    shape = [1] * self.order
                    shape[axis] = n
                    marginal = (term * w.view(*shape)).sum(axis)
                    weights = torch.ones_like(marginal)
                    for other in range(self.order - 1):
                        shape = [1] * (self.order - 1)
                        shape[other] = n
                        weights = weights * w.view(*shape)
                    result = result + (weights * marginal.square()).sum()
        return result


def joint_objective(model, batches, target, constraints, phase, progress=1.0, purity_weight=10.0):
    prediction, _, raw_orders, snapshots = model(batches)
    mse = nn.functional.mse_loss(prediction, target)
    total, records = mse, []
    for order, (stage, cfg, raw, snapshot, constraint) in enumerate(
            zip(model.stages, model.configs, raw_orders, snapshots, constraints), 1):
        p, gate = snapshot["entmax"], snapshot["gate"]
        strength = torch.sqrt((raw - raw.mean(0)).square().mean(0) + 1e-10).detach()
        sparse = (p.new_zeros(()) if int(snapshot["allowed"].sum()) <= 1
                  else (p / (0.02 + strength)).sum())
        pur_weight = 0.0 if cfg["disable_purity"] else purity_weight
        sp_weight = (cfg["sparse"] * progress if phase == "sparse" and not cfg["disable_sparse_loss"] else 0.0)
        purity = constraint(stage, gate) if pur_weight else mse.new_zeros(())
        total = total + pur_weight * purity + sp_weight * sparse
        records.append(dict(order=order, purity=float(purity.detach()), sparse=float(sparse.detach()),
                            purity_weight=pur_weight, sparse_weight=sp_weight))
    return total, dict(mse=float(mse.detach()), by_order=records)
