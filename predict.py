
import argparse
import ast
import csv
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import torch
from common import graph_batch, effect_names, score
from training_components import JointModel


def load_run(folder, device):
    path = Path(folder).resolve()
    checkpoint_path = path if path.is_file() else path / "best.pth.tar"
    cp = torch.load(checkpoint_path, map_location=device, weights_only=True)
    required = {"state_dict", "mean", "std", "highest_order", "variables", "feature_matrix", "bounds", "configs", "variant", "gate_mode"}
    if not required.issubset(cp):
        raise ValueError(f"Not a compatible joint checkpoint: missing {sorted(required - set(cp))}")
    features = cp["feature_matrix"]
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float32)
    model = JointModel(features.shape[1], cp["highest_order"], cp["configs"],
                       excluded_effects=cp.get("excluded_effects", []), variable_names=cp["variables"]).to(device)
    cp["excluded_effects"] = model.excluded_effects
    model.load_state_dict(cp["state_dict"], strict=True)
    model.variable_names = list(cp["variables"])
    model.feature_matrix = features
    bounds = cp["bounds"]
    model.input_bounds = np.asarray(bounds.detach().cpu() if torch.is_tensor(bounds) else bounds, dtype=float)
    for stage in model.stages:
        stage.layer.mix = 0.0 if cp["gate_mode"] == "uniform" else 1.0
        stage.layer.floor = 0.0
        stage.layer.temperature = 1.0 if cp["gate_mode"] == "uniform" else 0.4
    model.eval()
    return model, cp["mean"].to(device), cp["std"].to(device), cp


def predict_joint(model, x, mean, std, variables, feature_matrix, device, batch_size=1024):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(variables) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Prediction requires a nonempty finite array in checkpoint variable order.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if (getattr(model, "variable_names", list(variables)) != list(variables)
            or not np.array_equal(getattr(model, "feature_matrix", feature_matrix), feature_matrix)):
        raise ValueError("Prediction variables or features differ from the checkpoint.")
    predictions, terms = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batches = [graph_batch(x[start:start + batch_size], k, device, variables, feature_matrix)
                       for k in range(1, model.highest_order + 1)]
            pred, pieces, _, _ = model(batches)
            predictions.append((pred * std + mean).cpu().numpy().ravel().astype(float))
            terms.append((pieces * std).cpu().numpy().astype(float))
        intercept = float((sum(stage.intercept for stage in model.stages) * std + mean).item())
    full, pieces = np.concatenate(predictions), np.concatenate(terms)
    if not np.isfinite(full).all() or not np.isfinite(pieces).all():
        raise ValueError("The model produced nonfinite predictions.")
    np.testing.assert_allclose(full, pieces.sum(1) + intercept, rtol=2e-6, atol=2e-5)
    return full, pieces, intercept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--modelpath", required=True)
    parser.add_argument("--savepath", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--disable-cuda", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")
    model, mean, std, cp = load_run(args.modelpath, device)
    with Path(args.sample).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    variables = cp["variables"]
    x = np.asarray([[float(ast.literal_eval(row["composition"])[name]) for name in variables] for row in rows])
    full, terms, intercept = predict_joint(model, x, mean, std, variables, model.feature_matrix, device, args.batch_size)
    names = [name for order in range(1, cp["highest_order"] + 1) for name in effect_names(variables, order)]
    output = Path(args.savepath).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    records = [{**row, "prediction": float(full[i]), "intercept": intercept,
                **{f"gated_order_term_{j + 1:03d}": float(value) for j, value in enumerate(terms[i])}}
               for i, row in enumerate(rows)]
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = None
    if all(row.get("target", "").strip() for row in rows):
        metrics = score(np.asarray([float(row["target"]) for row in rows]), full)
    summary = dict(model=str(Path(args.modelpath).resolve()), variables=variables, effect_order=names,
                   excluded_effects=model.excluded_effects, exclusion_mode="exact_native_support",
                   terms_definition="Original-scale gated native terms before constant centering; no fANOVA",
                   target_mean_added_once=True, intercept=intercept, response_metrics=metrics)
    (output / "prediction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
