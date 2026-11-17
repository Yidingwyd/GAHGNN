
import argparse
import csv
import hashlib
import itertools
import json
from math import comb, sqrt
from pathlib import Path
import re
import sys
import warnings

sys.dont_write_bytecode = True

DEFAULT_POINTS = 51
DEFAULT_LOWER = None
DEFAULT_UPPER = None
DEFAULT_VARIABLE_BOUNDS = {}

# Also works when copied into a run folder below the model code folder.
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if all((candidate / name).is_file() for name in
           ("predict.py", "training_components.py", "model.py")):
        sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Keep this script in the model code folder or one of its run subfolders.")

import numpy as np
import torch

from common import effect_names, graph_batch, quadrature_axes
from predict import load_run


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require_continuous_checkpoint(checkpoint):
    measures = [checkpoint.get("integration_measure", "")]
    measures.extend(cfg.get("integration_measure", "") for cfg in checkpoint["configs"])
    discrete = (checkpoint.get("discrete_values") is not None or
                any(cfg.get("discrete_values") is not None for cfg in checkpoint["configs"]) or
                any("discrete" in str(measure).lower() for measure in measures))
    if discrete:
        raise ValueError("This exporter supports continuous joint checkpoints only. "
                         "A discrete checkpoint requires its discrete-domain exporter; "
                         "continuous interpolation would not preserve its measurement domain.")
    if checkpoint["gate_mode"] not in ("uniform", "entmax"):
        raise ValueError("Unsupported checkpoint gate_mode; expected 'uniform' or 'entmax'.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", "--modelpath", default=str(SCRIPT_DIR),
                        help="Joint run folder or best.pth.tar checkpoint; default: script folder")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS,
                        help="Points per variable including both endpoints (default: 51)")
    parser.add_argument("--lower", type=float, default=DEFAULT_LOWER,
                        help="Optional common lower limit for all variables")
    parser.add_argument("--upper", type=float, default=DEFAULT_UPPER,
                        help="Optional common upper limit for all variables")
    parser.add_argument("--bounds", nargs=3, action="append", default=[],
                        metavar=("VARIABLE", "LOWER", "UPPER"),
                        help="Override one variable, e.g. --bounds Al -1 1; repeat as needed")
    parser.add_argument("--output", "--savepath", default=None,
                        help="New output folder; default: RUN/effects_Npoints")
    parser.add_argument("--orders", nargs="+", type=int, choices=[1, 2, 3], default=None,
                        help="Optional subset of the trained orders to export")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--centering", choices=["grid", "training"], default="grid",
                        help="grid: subtract each exported grid's mean (default); training: retain the previous training-domain centering")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--allow-large", action="store_true",
                        help="Allow more than 5,000,000 total output rows")
    args = parser.parse_args()
    if args.points < 2 or args.batch_size < 1:
        parser.error("--points must be at least 2 and --batch-size must be positive.")
    return args


def selected_batch(x, support, device, features):
    values = torch.as_tensor(x, dtype=torch.float32, device=device)
    support = torch.as_tensor(support, dtype=torch.long, device=device)
    features = torch.as_tensor(features, dtype=torch.float32, device=device)
    count, order = values.shape[0], support.numel()
    weights = values[:, support].reshape(-1, 1)
    node_features = features[support].unsqueeze(0).expand(count, -1, -1).reshape(-1, features.shape[1])
    graph_idx = torch.arange(count, device=device).repeat_interleave(order)
    material_idx = torch.arange(count, device=device)
    comp_idx = support.unsqueeze(0).expand(count, -1).reshape(-1)
    return weights, node_features, comp_idx, graph_idx, material_idx


def evaluate_term(model, x, support, gate, std, device, batch_size):
    if gate == 0.0:
        return np.zeros(len(x), dtype=float)
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = selected_batch(x[start:start + batch_size], support, device, model.feature_matrix)
            raw = model(batch)[2][:, 0]
            values = raw * gate * std
            result.append(values.cpu().numpy().astype(float))
    values = np.concatenate(result)
    if not np.isfinite(values).all():
        raise ValueError("The model produced nonfinite effect values.")
    return values


def stage_gates(model, order, bounds, device):
    midpoint = bounds.mean(axis=1)[None, :]
    with torch.inference_mode():
        model(graph_batch(midpoint, order, device, model.variable_names, model.feature_matrix))
    return model.layer.last_gate.detach().cpu().numpy().astype(float).copy()


def term_center(model, support, gate, std, order, bounds, device, batch_size):
    if gate == 0.0:
        return 0.0
    n = [24, 16, 10][order - 1]
    axes, weights = quadrature_axes(n, bounds)
    indices = np.asarray(list(itertools.product(range(n), repeat=order)), dtype=int)
    x = np.repeat(bounds.mean(axis=1)[None, :], len(indices), axis=0)
    for column, variable in enumerate(support):
        x[:, variable] = axes[indices[:, column], variable]
    values = evaluate_term(model, x, support, gate, std, device, batch_size)
    return float(np.prod(weights[indices], axis=1) @ values)


def export_bounds(args, variables, training_bounds):
    bounds = np.array(training_bounds, dtype=float, copy=True)
    if args.lower is not None:
        bounds[:, 0] = args.lower
    if args.upper is not None:
        bounds[:, 1] = args.upper
    overrides = dict(DEFAULT_VARIABLE_BOUNDS)
    for name, lower, upper in args.bounds:
        try:
            overrides[name] = (float(lower), float(upper))
        except ValueError as exc:
            raise ValueError(f"Bounds for {name!r} must be numeric.") from exc
    for name, pair in overrides.items():
        if name not in variables:
            raise ValueError(f"Unknown variable {name!r}; model variables: {variables}")
        if len(pair) != 2:
            raise ValueError(f"Bounds for {name!r} must contain two numbers.")
        bounds[variables.index(name)] = pair
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError("Each range must have finite lower <= upper.")
    tolerance = 1e-12 * np.maximum(1.0, np.max(np.abs(training_bounds), axis=1))
    outside = ((bounds[:, 0] < training_bounds[:, 0] - tolerance) |
               (bounds[:, 1] > training_bounds[:, 1] + tolerance))
    if outside.any():
        warnings.warn("Export includes extrapolation outside training ranges for: " +
                      ", ".join(np.asarray(variables)[outside]))
    return bounds


def grid_batches(axes, support, midpoint, batch_size):
    combinations = itertools.product(*(axes[i] for i in support))
    while True:
        rows = list(itertools.islice(combinations, batch_size))
        if not rows:
            return
        coordinates = np.asarray(rows, dtype=float)
        x = np.repeat(midpoint[None, :], len(rows), axis=0)
        x[:, list(support)] = coordinates
        yield coordinates, x


def grid_values(model, support, gate, std, axes, device, batch_size):
    rows = int(np.prod([len(axes[i]) for i in support]))
    values = np.empty(rows, dtype=float)
    start = 0
    for _, x in grid_batches(axes, support, model.input_bounds.mean(axis=1), batch_size):
        batch_values = evaluate_term(model, x, support, gate, std, device, batch_size)
        values[start:start + len(x)] = batch_values
        start += len(x)
    return values


def centered_grid(values, training_center, mode):
    native = np.asarray(values, dtype=float) - training_center
    grid_mean_before = float(np.mean(native, dtype=np.float64))
    grid_removed = grid_mean_before if mode == "grid" else 0.0
    final = native - grid_removed
    if mode == "grid":
        # Remove residual floating-point roundoff as well as the learned offset.
        correction = float(np.mean(final, dtype=np.float64))
        final -= correction
        grid_removed += correction
    variance = float(np.var(final, ddof=0, dtype=np.float64))
    statistics = dict(training_center_removed=float(training_center),
                      grid_mean_before_center=grid_mean_before,
                      grid_mean_removed=float(grid_removed),
                      center_removed=float(training_center + grid_removed),
                      mean_after_center=float(np.mean(final, dtype=np.float64)),
                      variance=variance, standard_deviation=sqrt(variance))
    return final, statistics


def write_effect_csv(path, model, support, axes, values, value_column, batch_size):
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([model.variable_names[i] for i in support] + [value_column])
        for coordinates, _ in grid_batches(axes, support, model.input_bounds.mean(axis=1), batch_size):
            batch_values = values[count:count + len(coordinates)]
            writer.writerows([*point, float(value)] for point, value in zip(coordinates, batch_values))
            count += len(coordinates)
    return count


def reconstruction_check(joint, stages, term_specs, intercept, mean, std,
                         bounds, device, batch_size):
    points = np.random.default_rng(20261019).uniform(bounds[:, 0], bounds[:, 1],
                                                   size=(10, len(bounds)))
    reconstructed = np.full(len(points), intercept, dtype=float)
    for model, support, gate, center in term_specs:
        reconstructed += evaluate_term(model, points, support, gate, std, device, batch_size) - center
    original_subset = np.full(len(points), mean, dtype=float)
    with torch.inference_mode():
        for model, _, _, checkpoint in stages:
            batch = graph_batch(points, checkpoint["order"], device,
                                joint.variable_names, joint.feature_matrix)
            original_subset += (model(batch)[0] * std).cpu().numpy().ravel().astype(float)
    np.testing.assert_allclose(reconstructed, original_subset, rtol=2e-6, atol=2e-5,
                               err_msg="Centered native effects do not reconstruct the selected-order model.")
    return dict(status="passed", points=10, random_seed=20261019,
                comparison="Centered exported native terms plus total intercept versus original selected-stage forward",
                max_absolute_error=float(np.max(np.abs(reconstructed - original_subset))),
                rtol=2e-6, atol=2e-5)


def main():
    args = parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")
    run = Path(args.run).resolve()
    checkpoint_path = run if run.is_file() else run / "best.pth.tar"
    hashes = {str(checkpoint_path): digest(checkpoint_path)}
    joint, mean_tensor, std_tensor, checkpoint = load_run(run, device)
    require_continuous_checkpoint(checkpoint)
    variables = joint.variable_names
    excluded = {tuple(group) for group in joint.excluded_effects}
    features = joint.feature_matrix
    training_bounds = np.asarray(joint.input_bounds, dtype=float)
    if (training_bounds.shape != (len(variables), 2) or not np.isfinite(training_bounds).all()
            or np.any(training_bounds[:, 1] < training_bounds[:, 0])):
        raise ValueError("Checkpoint has invalid per-variable training bounds.")
    mean, std = float(mean_tensor.item()), float(std_tensor.item())
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("Checkpoint target mean and standard deviation must be finite, with std > 0.")
    for stage in joint.stages:
        stage.variable_names = variables
        stage.feature_matrix = features
        stage.input_bounds = training_bounds
    stages = [(stage, mean, std, {"order": order})
              for order, stage in enumerate(joint.stages, start=1)]
    if args.orders is not None:
        missing = set(args.orders) - set(range(1, joint.highest_order + 1))
        if missing:
            raise ValueError(f"Requested orders were not trained: {sorted(missing)}")
        stages = [stage for stage in stages if stage[3]["order"] in args.orders]
    selected_orders = [stage[3]["order"] for stage in stages]
    is_full_model = selected_orders == list(range(1, joint.highest_order + 1))
    bounds = export_bounds(args, variables, training_bounds)
    axes = [np.linspace(lower, upper, args.points, endpoint=True) for lower, upper in bounds]
    has_order2 = any(stage[3]["order"] == 2 for stage in stages)
    if has_order2 and (args.points ** 2 + 1 > 1_048_576 or args.points + 1 > 16_384):
        raise ValueError("Order-2 grids exceed Excel's row/column limits. Reduce --points.")
    total_rows = sum(comb(len(variables), s[3]["order"]) *
                     args.points ** s[3]["order"] for s in stages)
    if total_rows > 5_000_000 and not args.allow_large:
        raise ValueError(f"Export would contain {total_rows:,} rows. Reduce --points/--orders, "
                         "or explicitly use --allow-large.")
    if has_order2:
        from export_effects_xlsx import write_effect_xlsx

    root = run.parent if run.is_file() else run
    output = Path(args.output).resolve() if args.output else root / f"effects_{args.points}points"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty output: {output}. Use a new --output.")
    output.mkdir(parents=True, exist_ok=True)
    value_column = "effect_value"
    while value_column in variables:
        value_column = "_" + value_column
    records, intercepts, raw_stage_intercepts, term_specs = [], {}, {}, []
    for model, mean, std, checkpoint in stages:
        order = checkpoint["order"]
        supports = list(itertools.combinations(range(len(variables)), order))
        names = effect_names(variables, order)
        gates = stage_gates(model, order, training_bounds, device)
        stage_dir = output / f"order{order}"
        stage_dir.mkdir()
        centers = []
        for index, (support, name, gate) in enumerate(zip(supports, names, gates), start=1):
            training_center = term_center(model, support, float(gate), std, order, training_bounds,
                                          device, args.batch_size)
            raw_values = grid_values(model, support, float(gate), std, axes, device, args.batch_size)
            values, statistics = centered_grid(raw_values, training_center, args.centering)
            centers.append(statistics["center_removed"])
            term_specs.append((model, support, float(gate), statistics["center_removed"]))
            label = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")[:80] or "effect"
            extension = "xlsx" if order == 2 else "csv"
            filename = f"effect_{index:03d}_{label}.{extension}"
            rows = len(values)
            if order == 2:
                write_effect_xlsx(stage_dir / filename, [variables[i] for i in support],
                                  [axes[i] for i in support], values, value_column)
            else:
                write_effect_csv(stage_dir / filename, model, support, axes, values, value_column, args.batch_size)
            record = dict(order=order, effect=name, variables=[variables[i] for i in support],
                          prior_excluded=tuple(variables[i] for i in support) in excluded,
                          gate=float(gate), **statistics, rows=rows, file=f"order{order}/{filename}",
                          matrix_row_variable=variables[support[0]] if order == 2 else "",
                          matrix_column_variable=variables[support[1]] if order == 2 else "")
            records.append(record)
            print(f"order={order} effect={name} rows={rows} gate={gate:.8g} "
                  f"mean={statistics['mean_after_center']:.3g} variance={statistics['variance']:.8g}", flush=True)
        raw_stage_intercepts[str(order)] = float((model.intercept * std).item())
        intercepts[str(order)] = float(raw_stage_intercepts[str(order)] + sum(centers))
    total_intercept = float(mean + sum(intercepts.values()))
    reconstruction = reconstruction_check(joint, stages, term_specs, total_intercept, mean, std,
                                          bounds, device, args.batch_size)
    for model, _, _, stage_checkpoint in stages:
        # Selected-support calls have a singleton gate; restore the full-candidate diagnostic cache.
        stage_gates(model, stage_checkpoint["order"], training_bounds, device)
    if any(digest(path) != saved for path, saved in hashes.items()):
        raise AssertionError("A checkpoint changed during export.")
    total_native_variance = float(sum(record["variance"] for record in records))
    for record in records:
        record["sum_native_term_variances"] = total_native_variance
        record["native_term_variance_fraction"] = (record["variance"] / total_native_variance
                                                   if total_native_variance > 0 else None)
        record["native_term_variance_percent"] = (100 * record["variance"] / total_native_variance
                                                  if total_native_variance > 0 else None)
    with (output / "effect_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "variables": json.dumps(record["variables"], ensure_ascii=False)})
    metadata = dict(status="completed", source=str(run), checkpoint_sha256=hashes,
                    checkpoint=str(checkpoint_path), joint_protocol=True,
                    highest_trained_order=joint.highest_order, selected_orders=selected_orders,
                    model_scope="full_joint_model" if is_full_model else "selected_order_submodel",
                    full_model_exported=is_full_model, no_refit=True,
                    variables=variables, training_bounds=training_bounds.tolist(),
                    excluded_effects=joint.excluded_effects, exclusion_mode="exact_native_support",
                    export_bounds=bounds.tolist(), points_per_variable=args.points,
                    endpoint=True, effect_value_column=value_column,
                    grid_order="Cartesian product; last support variable varies fastest",
                    total_rows=total_rows, device=str(device),
                    interpretation="Native gated model terms with constant-mean centering; no post-hoc fANOVA",
                    centering_mode=args.centering,
                    centering_reference=("Equal-weight mean of this exported Cartesian grid" if args.centering == "grid"
                                         else "Saved training bounds, uniform product measure; fixed across export ranges"),
                    centering_quadrature_nodes={"1": 24, "2": 16, "3": 10},
                    variance_ddof=0,
                    variance_definition="Population variance of each exported effect on its own equal-weight endpoint-inclusive grid",
                    sum_native_term_variances=total_native_variance,
                    variance_fraction_definition="Each exported native-term variance divided by the sum of exported native-term variances; not a strict response-variance decomposition",
                    response_variance_computed=False,
                    variance_caution="Term covariances are not included. The sum of native-term variances need not equal full-response variance.",
                    order2_format="XLSX with long and matrix sheets; first support variable is rows, second is columns",
                    normalization="One shared full-response training std rescales every order; the shared response mean is added exactly once",
                    target_mean=mean, target_std=std, target_mean_added_once=True,
                    stage_intercepts_before_centering=raw_stage_intercepts,
                    stage_intercepts=intercepts,
                    stage_intercepts_definition="Each selected stage intercept times shared std plus that stage's removed native-term constants; excludes the shared response mean",
                    total_intercept_definition="Shared response mean + selected stage intercepts times shared std + all selected native-term centering constants",
                    total_intercept_for_exported_stages=total_intercept,
                    reconstruction_check=reconstruction, effects=records)
    (output / "export_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"EXPORT_COMPLETED {output} total_rows={total_rows}", flush=True)


if __name__ == "__main__":
    main()
