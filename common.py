import ast
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import torch


def normalize_excluded_effects(groups, variable_names, highest_order):
    """Validate exact-support exclusions and return names in canonical order."""
    indices = {name: index for index, name in enumerate(variable_names)}
    unique = set()
    for group in groups or []:
        if isinstance(group, str) or not group:
            raise ValueError("Each excluded effect must be a nonempty list of variable names.")
        if any(name not in indices for name in group):
            unknown = [name for name in group if name not in indices]
            raise ValueError(f"Unknown variables in excluded effect: {unknown}; available: {list(variable_names)}.")
        if len(set(group)) != len(group):
            raise ValueError(f"An excluded effect cannot repeat a variable: {group}.")
        if len(group) > highest_order:
            raise ValueError(f"Excluded effect {group} exceeds the highest trained order ({highest_order}).")
        unique.add(tuple(sorted(indices[name] for name in group)))
    return [[variable_names[index] for index in support]
            for support in sorted(unique, key=lambda support: (len(support), support))]


def effect_names(variable_names, order):
    combos = list(itertools.combinations(variable_names, order))
    names = ["_".join(combo) for combo in combos]
    if len(set(names)) != len(names):
        return [json.dumps(combo, ensure_ascii=False, separators=(",", ":")) for combo in combos]
    return names


def load_embedding(path, variables=None):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or not value:
        raise ValueError("Embedding JSON must be a nonempty object mapping each variable name to a numeric feature vector.")
    names = list(value) if variables is None else list(variables)
    if not names or len(set(names)) != len(names):
        raise ValueError("Variable names must be nonempty and unique.")
    if set(value) != set(names):
        missing = sorted(set(names) - set(value))
        extra = sorted(set(value) - set(names))
        raise ValueError(f"Embedding keys and requested variables must match exactly; missing={missing}, extra={extra}.")
    try:
        features = np.asarray([value[name] for name in names], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every embedding value must be a numeric feature vector of the same length.") from exc
    if features.ndim != 2 or features.shape[1] < 1 or not np.isfinite(features).all():
        raise ValueError("Embedding vectors must be finite, one-dimensional vectors with a shared positive feature length.")
    return names, features


def input_bounds(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite n-by-p feature array.")
    return np.column_stack((x.min(axis=0), x.max(axis=0)))


def quadrature_axes(n, bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] < 1 or not np.isfinite(bounds).all() or np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError("Bounds must contain one finite [lower, upper] pair for each variable.")
    q, w = np.polynomial.legendre.leggauss(n)
    lower, upper = bounds[:, 0], bounds[:, 1]
    axes = (lower + upper)[None, :] / 2 + q[:, None] * (upper - lower)[None, :] / 2
    return axes, w / 2


def read_data(path, variable_names):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No data rows found in {path}.")
    if not {"composition", "target"}.issubset(rows[0]):
        raise ValueError("CSV must contain 'composition' and 'target' columns.")
    parsed = []
    for row_number, row in enumerate(rows, start=2):
        try:
            composition = ast.literal_eval(row["composition"])
            parsed.append([float(composition[name]) for name in variable_names])
        except (ValueError, SyntaxError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid composition or missing variable at CSV row {row_number}.") from exc
    x = np.asarray(parsed, dtype=float)
    y = np.asarray([float(row["target"]) for row in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Inputs and targets must be finite numeric values.")
    return rows, x, y


def graph_batch(x, size, device, variable_names, feature_matrix):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    features = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    n_variables = len(variable_names)
    if x.ndim != 2 or x.shape[1] != n_variables or features.ndim != 2 or features.shape[0] != n_variables:
        raise ValueError("Input columns and embedding rows must match the selected variable list.")
    if size < 1 or size > n_variables:
        raise ValueError(f"Effect order must be between 1 and the number of variables ({n_variables}).")
    members = torch.tensor(list(itertools.combinations(range(n_variables), size)), dtype=torch.long, device=device)
    count, edges = len(x), len(members)
    weights = x[:, members].reshape(-1, 1)
    node_features = features[members].unsqueeze(0).expand(count, -1, -1, -1).reshape(-1, features.shape[1])
    graph_idx = torch.arange(count * edges, device=device).repeat_interleave(size)
    material_idx = torch.arange(count, device=device).repeat_interleave(edges)
    comp_idx = members.unsqueeze(0).expand(count, -1, -1).reshape(-1)
    return weights, node_features, comp_idx, graph_idx, material_idx


def tensor_inputs(x, device, variable_names, feature_matrix, max_order=3):
    return [graph_batch(x, k, device, variable_names, feature_matrix) for k in range(1, min(max_order, len(variable_names)) + 1)]


def score(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    sse = float(np.sum((p - y) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return dict(r2=None if sst < 1e-12 else 1 - sse / sst, rmse=float(np.sqrt(sse / len(y))), mae=float(np.mean(abs(p - y))), max_abs_error=float(np.max(abs(p - y))), sse=sse, sst=sst)


class PurityConstraint:
    def __init__(self, device, bounds, variable_names, feature_matrix):
        self.device = device
        self.bounds = np.asarray(bounds, dtype=float)
        self.variable_names = list(variable_names)
        self.feature_matrix = np.asarray(feature_matrix, dtype=float)
        self.axes_by_order = {}
        self.batches = {}
        for order, n in ((1, 16), (2, 12), (3, 8)):
            if order > len(self.variable_names):
                continue
            axes, weights = quadrature_axes(n, self.bounds)
            self.axes_by_order[order] = (axes, weights)
            combos = list(itertools.combinations(range(len(self.variable_names)), order))
            supports = []
            for combo in combos:
                meshes = np.meshgrid(*(axes[:, i] for i in combo), indexing="ij")
                points = np.repeat(self.bounds.mean(axis=1)[None, :], n ** order, axis=0)
                for i, mesh in zip(combo, meshes):
                    points[:, i] = mesh.ravel()
                supports.append((combo, graph_batch(points, order, device, self.variable_names, self.feature_matrix)))
            self.batches[order] = supports

    def __call__(self, model):
        penalty = model.intercept.new_zeros(())
        for order, supports in self.batches.items():
            _, weights = self.axes_by_order[order]
            for index, (combo, batch) in enumerate(supports):
                term = model(batch)[1][:, index]
                shaped = term.reshape(*([len(weights)] * order))
                for axis in range(order):
                    weight_shape = [1] * order
                    weight_shape[axis] = len(weights)
                    conditional = (shaped * torch.as_tensor(weights, dtype=term.dtype, device=term.device).view(*weight_shape)).sum(axis)
                    remaining_shape = [len(weights)] * (order - 1)
                    marginal_weights = torch.ones(remaining_shape, dtype=term.dtype, device=term.device)
                    for remaining_axis in range(order - 1):
                        marginal_weights = marginal_weights * torch.as_tensor(weights, dtype=term.dtype, device=term.device).view(*([1] * remaining_axis), len(weights), *([1] * (order - remaining_axis - 2)))
                    penalty = penalty + (marginal_weights * conditional.square()).sum()
        return penalty


def assess(model, x, order, mean, std, device, variable_names, feature_matrix):
    model.eval()
    predictions, terms = [], []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            p, t, _ = model(graph_batch(x[start:start + 1024], order, device, variable_names, feature_matrix))
            predictions.append((p * std + mean).cpu().numpy().ravel())
            terms.append((t * std).cpu().numpy())
    return np.concatenate(predictions).astype(float), np.concatenate(terms).astype(float)
