
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from common import graph_batch, input_bounds, load_embedding, read_data, score, normalize_excluded_effects
from training_components import JointModel, OrderConstraint, joint_objective, order_config
from predict import predict_joint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_digest(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode("utf-8"))
        result.update(str(tuple(value.shape)).encode("ascii"))
        result.update(str(value.dtype).encode("ascii"))
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def phase_lengths(epochs, variant):
    if variant == "no_dense":
        return 0, epochs
    if variant == "no_sparse_stage":
        return epochs, 0
    dense = max(1, min(epochs - 1, round(epochs * 6 / 11)))
    return dense, epochs - dense


def set_phase(model, phase, progress, uniform_final=False):
    for stage, cfg in zip(model.stages, model.configs):
        stage.layer.mix = 0.0 if phase == "dense" or uniform_final else 1.0
        stage.layer.floor = 0.05 * (1 - progress) if phase == "sparse" else 0.0
        stage.layer.temperature = 1 + (cfg["temp"] - 1) * progress if phase == "sparse" else 1.0
        if phase == "polish" and not uniform_final:
            stage.layer.temperature = cfg["temp"]


def optimizer(model, phase):
    groups = []
    for stage, cfg in zip(model.stages, model.configs):
        functions, scores = [], []
        for name, parameter in stage.named_parameters():
            if parameter.requires_grad:
                (scores if ".ent_gate." in name or ".selector_skip." in name or name.endswith(".alpha") else functions).append(parameter)
        groups.extend([dict(params=functions, lr=0.001 if phase == "dense" else 0.0003),
                       dict(params=scores, lr=0.001 if phase == "dense" else cfg["score_lr"])])
    return torch.optim.AdamW(groups, betas=(0.9, 0.999 if phase == "dense" else 0.95), weight_decay=0.0)


def gate_values(model, batches):
    with torch.no_grad():
        _, _, _, snapshots = model(batches)
    return [snapshot["gate"].detach().cpu().tolist() for snapshot in snapshots]


def train(args):
    root = Path(args.savepath).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty run output: {root}")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cpu" if args.disable_cuda else getattr(args, "device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --disable-cuda for CPU execution.")
    requested = [name.strip() for name in args.variables.split(",")] if args.variables else None
    variables, features = load_embedding(args.embedding, requested)
    if args.size > len(variables):
        raise ValueError("Highest order cannot exceed the number of variables.")
    excluded_effects = normalize_excluded_effects(getattr(args, "excluded_effects", []), variables, args.size)
    train_rows, x, y = read_data(args.train_data, variables)
    val_rows, vx, vy = read_data(args.val_data, variables)
    if args.batch_size < max(len(x), len(vx)):
        raise ValueError("This protocol uses full batches; increase --batch-size to fit all rows.")
    if len(y) < 2 or np.std(y) <= 1e-12:
        raise ValueError("Training responses must have nonzero variance and at least two rows.")
    source_hashes = {"train": digest(args.train_data), "val": digest(args.val_data), "embedding": digest(args.embedding)}
    bounds = input_bounds(x)
    configs = [order_config(k, args.ablation) for k in range(1, args.size + 1)]
    model = JointModel(features.shape[1], args.size, configs,
                       excluded_effects=excluded_effects, variable_names=variables).to(device)
    model.variable_names, model.feature_matrix, model.input_bounds = variables, features, bounds
    initial_digest = parameter_digest(model)
    initial_stage_digests = [parameter_digest(stage) for stage in model.stages]
    code_dir = Path(__file__).resolve().parent
    source_code_sha256 = {name: digest(code_dir / name) for name in
                         ("common.py", "model.py", "model_base.py", "roost_segments.py",
                          "training_components.py", "train_joint.py", "main.py", "predict.py")}
    train_batches = [graph_batch(x, k, device, variables, features) for k in range(1, args.size + 1)]
    val_batches = [graph_batch(vx, k, device, variables, features) for k in range(1, args.size + 1)]
    gate_batches = [graph_batch(x[:1], k, device, variables, features) for k in range(1, args.size + 1)]
    constraints = [OrderConstraint(k, device, cfg, bounds, variables, features) for k, cfg in enumerate(configs, 1)]
    yt = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)
    mean, std = yt.mean(), yt.std()
    target = (yt - mean) / std
    vtarget = (torch.as_tensor(vy, dtype=torch.float32, device=device).reshape(-1, 1) - mean) / std
    dense, sparse = phase_lengths(args.epochs, args.ablation)
    root.mkdir(parents=True, exist_ok=True)
    history, selected_epochs = [], {}
    started, global_epoch = time.perf_counter(), 0
    for phase, epochs in (("dense", dense), ("sparse", sparse)):
        if not epochs:
            continue
        opt = optimizer(model, phase)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                   eta_min=3e-5 if phase == "dense" else 1e-5)
        best, best_state, best_epoch = float("inf"), None, None
        for epoch in range(epochs):
            progress = min(1.0, (epoch + 1) / (epochs * 0.6))
            set_phase(model, phase, progress)
            pur = 10 * min(1.0, (epoch + 1) / (epochs * 0.2)) if phase == "dense" else 10.0
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, metrics = joint_objective(model, train_batches, target, constraints, phase, progress, pur)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective in {phase}, epoch {epoch}.")
            loss.backward()
            for stage in model.stages:
                torch.nn.utils.clip_grad_norm_(stage.parameters(), 10.0)
            opt.step()
            schedule.step()
            row = dict(phase=phase, epoch=epoch, global_epoch=global_epoch,
                       training_objective=float(loss.detach()), training=metrics,
                       gates=gate_values(model, gate_batches), gate_snapshot="after_optimizer_step")
            if epoch % 10 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    validation, val_metrics = joint_objective(model, val_batches, vtarget, constraints, phase, 1.0, 10.0)
                row.update(validation_objective=float(validation), validation=val_metrics)
                if (phase == "dense" or progress == 1.0) and float(validation) < best:
                    best, best_state, best_epoch = float(validation), copy.deepcopy(model.state_dict()), epoch
                if epoch % 200 == 0 or epoch == epochs - 1:
                    print(json.dumps(row), flush=True)
            if phase == "sparse" or epoch % 10 == 0 or epoch == epochs - 1:
                history.append(row)
            global_epoch += 1
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint was selected.")
        model.load_state_dict(best_state)
        selected_epochs[phase] = dict(epoch=best_epoch, validation_objective=best)
        history.append(dict(phase=phase + "_selected", epoch=best_epoch, event="best_checkpoint_restored",
                            validation_objective=best, gates=gate_values(model, gate_batches)))
    uniform = args.ablation == "no_sparse_stage"
    set_phase(model, "polish", 1.0, uniform_final=uniform)
    for name, parameter in model.named_parameters():
        if ".ent_gate." in name or ".selector_skip." in name or name.endswith(".alpha"):
            parameter.requires_grad_(False)
    fixed_gates = gate_values(model, gate_batches)
    with torch.no_grad():
        value, _ = joint_objective(model, val_batches, vtarget, constraints, "polish", 1.0, 10.0)
    best, best_state, best_block = float(value), copy.deepcopy(model.state_dict()), -1
    opt = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad], lr=0.5, max_iter=10,
             history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-10)
    for block in range(args.polish_steps):
        def closure():
            opt.zero_grad(set_to_none=True)
            value, _ = joint_objective(model, train_batches, target, constraints, "polish", 1.0, 10.0)
            value.backward()
            return value
        opt.step(closure)
        with torch.no_grad():
            value, record = joint_objective(model, val_batches, vtarget, constraints, "polish", 1.0, 10.0)
        current_gates = gate_values(model, gate_batches)
        if current_gates != fixed_gates:
            raise AssertionError("Frozen selector gates changed during refinement.")
        if float(value) < best:
            best, best_state, best_block = float(value), copy.deepcopy(model.state_dict()), block
        row = dict(phase="fixed_gate_lbfgs", epoch=block, validation_objective=float(value),
                   validation=record, gates=current_gates)
        history.append(row)
        if block % 20 == 0 or block == args.polish_steps - 1:
            print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    gates = gate_values(model, gate_batches)
    if gates != fixed_gates:
        raise AssertionError("Refinement changed a selector gate.")
    history.append(dict(phase="final", gates=gates, event="selected_final_checkpoint"))
    cp = dict(state_dict=model.state_dict(), mean=mean.detach().cpu(), std=std.detach().cpu(),
              highest_order=args.size, variables=variables, feature_matrix=torch.as_tensor(features),
              bounds=torch.as_tensor(bounds, dtype=torch.float64), configs=configs, variant=args.ablation,
              gate_mode="uniform" if uniform else "entmax", joint_protocol=True, seed=args.seed,
              group_penalty_removed=True,
              excluded_effects=excluded_effects, exclusion_mode="exact_native_support",
              source_hashes=source_hashes, source_code_sha256=source_code_sha256, gates=gates,
              ablation=args.ablation, initial_parameter_sha256=initial_digest,
              initial_parameter_sha256_by_order=initial_stage_digests,
              train_data=str(Path(args.train_data).resolve()), val_data=str(Path(args.val_data).resolve()))
    torch.save(cp, root / "best.pth.tar")
    (root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    response_metrics = {}
    for name, rows, inputs, truth in (("train", train_rows, x, y), ("val", val_rows, vx, vy)):
        prediction, _, intercept = predict_joint(model, inputs, mean, std, variables, features, device)
        response_metrics[name] = score(truth, prediction)
        records = [{**row, "prediction": float(prediction[i]), "intercept": intercept} for i, row in enumerate(rows)]
        with (root / f"{name}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    assert source_hashes == {"train": digest(args.train_data), "val": digest(args.val_data), "embedding": digest(args.embedding)}
    manifest = dict(status="completed", joint_protocol=True, group_penalty_removed=True,
                    variant=args.ablation, seed=args.seed,
                    highest_order=args.size, variables=variables, feature_dim=features.shape[1],
                    excluded_effects=excluded_effects, exclusion_mode="exact_native_support",
                    feature_matrix=features.tolist(), bounds=bounds.tolist(),
                    normalization="One full-response training mean and sample standard deviation shared by every order; response mean added once",
                    target_mean=float(mean), target_std=float(std), original_train_data=str(Path(args.train_data).resolve()),
                    original_val_data=str(Path(args.val_data).resolve()), source_hashes=source_hashes,
                    rows=[len(x), len(vx)], training_target="full_response_for_all_orders_simultaneously",
                    true_effect_labels_used=False, sequential_residuals_used=False, pruning_used=False,
                    epochs=args.epochs, dense_epochs=dense, sparse_epochs=sparse,
                    polish_steps=args.polish_steps, polish_max_iter_per_step=10, phase_selected_epochs=selected_epochs,
                    polish_selected_block=best_block, gate_mode=cp["gate_mode"], gates=gates, configs=configs,
                    response_metrics=response_metrics, seconds=time.perf_counter() - started,
                    checkpoint_sha256=digest(root / "best.pth.tar"), device=str(device))
    manifest.update(initial_parameter_sha256=initial_digest,
                    initial_parameter_sha256_by_order=initial_stage_digests,
                    source_code_sha256=source_code_sha256)
    (root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "completed.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("JOINT_FINISHED " + json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest
