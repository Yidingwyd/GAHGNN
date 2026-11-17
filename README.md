# GAHGNN

Generalized Additive Hypergraph Neural Network

For questions about the code, please contact [Yiding Wang](mailto:yidingwyd@163.com).

## Contents

- [Brief review](#brief-review)
- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Input of GAHGNN](#input-of-gahgnn)
  - [Train a GAHGNN model](#train-a-gahgnn-model)
  - [Predict with a trained model](#predict-with-a-trained-model)
  - [Export learned effects](#export-learned-effects)
- [Useful utilities](#useful-utilities)
- [An example workflow for using GAHGNN](#an-example-workflow-for-using-gahgnn)
- [Example datasets and checkpoints](#example-datasets-and-checkpoints)
- [Acknowledgement](#acknowledgement)

## Brief review

GAHGNN is an intrinsically interpretable regression model. It represents variables as nodes, main effects as self-loops, pairwise interactions as edges, and higher-order interactions as hyperedges. The response is a sum of a baseline and learned effect functions. Each function depends only on the variables in its corresponding self-loop, edge, or hyperedge. Effect functions of the same order share network parameters, and entmax gates can set individual effects to zero during training.


## How to cite

Our paper is submitted.

## Prerequisites

numpy
torch
entmax
torch-scatter
pandas
openpyxl

## Usage

Run the commands below from the repository root. Use a new or empty output directory for each run.

### Input of GAHGNN

Training requires two CSV files (training and validation) and one JSON file defining the node features. You supply the data split; the program does not split the data automatically.

**Training and validation CSV.** Each row has a `composition` field containing a quoted dictionary of variable names and numeric values, and a numeric `target` field. `material_id` is optional but useful for matching the results to individual samples. The name `composition` is historical: the fields can represent any scalar variables, not just chemical compositions.

```csv
material_id,composition,target
sample_001,"{'Al': -1.0, 'Cr': 0.0, 'Mo': 0.5}",0.42
sample_002,"{'Al': 0.5, 'Cr': -0.5, 'Mo': 1.0}",1.31
```

**Node-feature JSON.** Use the same variable names as in the CSV and give each variable a numeric feature vector of the same length. These vectors describe the variables and are shared across samples; the values in `composition` vary from sample to sample. The vectors can be one-hot variable identifiers or physical descriptors.

```json
{
  "Al": [1, 0, 0],
  "Cr": [0, 1, 0],
  "Mo": [0, 0, 1]
}
```



### Train a GAHGNN model

```sh
python main.py --train_data TRAIN.csv --val_data VAL.csv --embedding EMBEDDING.json --savepath RUN_DIR --size 3
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--size` | Required | `1`: main effects only; `2`: main and pairwise effects; `3`: main, pairwise, and three-way effects. It cannot exceed the number of variables. |
| `--seed` | `11` | Random seed. |
| `--epochs` | `1100` | Total AdamW epochs, normally split into about 6/11 dense pretraining and 5/11 sparse selection. |
| `--polish-steps` | `40` | Number of L-BFGS refinement blocks, each with up to 10 internal iterations. Use `0` to skip refinement. |
| `--batch-size` | `1024` | This implementation uses full batches.|
| `--device` | `cuda` | Use `--device cpu` (or `--disable-cuda`) without CUDA. |
| `--exclude-effect` | None | Repeatable option for fixing a specified effect to zero. |
| `--ablation` | `full` | Selects the full training protocol or an ablation configuration. |

The default training protocol has three phases, each operating on all requested orders:

1. **Dense pretraining (600 epochs).** All allowed effects within an order receive equal gate weights while their functions learn. The conditional-mean purification penalty is gradually enabled. It penalizes nonzero main-effect means and lower-order content in interaction functions, using quadrature over the training bounds. It is a soft constraint, not a guarantee of exact orthogonality.
2. **Sparse selection (500 epochs).** Effect gates are learned with entmax (fixed $\alpha=1.5$). During the first 60% of this phase, the gate temperature decreases from 1 to 0.4, the uniform component decreases from 5% to zero, and the adaptive gate-sparsity penalty is ramped up. After the transition, entmax can set gates exactly to zero; no post-training pruning is needed.

The prediction loss is MSE on the standardized full response. The two regularizers above are used where applicable. Model checkpoints are selected using the supplied validation set, so use a separate test set for an independent assessment.

The training command writes the following files to `--savepath`:

| File | Contents |
| --- | --- |
| `best.pth.tar` | Selected model, variable order and features, training bounds, response scaling, and gate configuration. |
| `history.json` | Objectives and gate snapshots, every 10 epochs during dense pretraining (plus the last epoch) and every epoch during sparse selection, with additional selection and refinement records. |
| `train_predictions.csv`, `val_predictions.csv` | Full-response predictions in the original target units. |
| `run_manifest.json` | Run configuration, selected epochs, final gates, and train/validation metrics including RMSE and R². |

### Predict with a trained model

```sh
python predict.py --sample QUERY.csv --modelpath RUN_DIR --savepath PREDICTION_DIR
```

- `--modelpath` accepts either a run directory containing `best.pth.tar` or the checkpoint file itself. Provide the same `composition` format as for training, with every model variable present. `target` is optional. The checkpoint stores the variable features. Add `--disable-cuda` to force CPU inference.

- `predictions.csv` contains a prediction, an intercept, and the native model terms (`gated_order_term_001`, etc.) for each requested row. The intercept plus the sum of the terms gives the prediction. These terms are in the original target units and have not been mean-centered.
- `prediction_summary.json` lists the effect corresponding to each term column in `effect_order`. If every row has a target, the summary also reports full-response metrics, including RMSE and R².

### Export learned effects

Use `export_effects.py` to extract the native effect functions over a grid, rather than evaluating only the rows of a sample file. This does not retrain the model.

```sh
python export_effects.py --run RUN_DIR --points 51 --output EFFECT_DIR
```

- By default, each variable ranges from its minimum to its maximum in the **training set**. `--points 51` uses 51 equally spaced points per variable, including both endpoints. An order-$k$ effect is evaluated at $51^k$ combinations.
- Use `--lower` and `--upper` to set common limits, or repeat `--bounds VARIABLE MIN MAX` to override individual variables. For example, `--bounds Al -1 1 --bounds Cr -0.5 1.2` changes only Al and Cr. Bounds outside the training range are extrapolations.
- Omit `--orders` to export every trained order, or use, for example, `--orders 1 2`. Large grids grow quickly; the exporter requires `--allow-large` when the combined output exceeds 5,000,000 grid rows.
- The default `--centering grid` subtracts each effect's own grid average. `--centering training` instead uses a fixed uniform reference over the training bounds, estimated by quadrature. This is useful when comparing exports with different query bounds; it is not an average over the observed training rows.

| Output | Contents |
| --- | --- |
| `order1/`, `order3/` | One CSV per selected main or three-way effect. |
| `order2/` | One XLSX per pair. The `long` sheet lists one row per grid point. In `matrix`, the first variable supplies the row coordinates and the second supplies the column coordinates. |
| `effect_summary.csv` | Gate, centering offset, population variance and standard deviation on each exported grid, and the term's share of the summed variances of the exported terms. |
| `export_metadata.json` | Checkpoint, grid settings, centering reference, and adjusted intercept for the selected orders. |

The exported values are **the model's own gated effects**, not a post-hoc fANOVA. The variance shares describe the extracted terms; they do not account for covariance and should not be read as an exact decomposition of the full response variance. This script does not calculate DGSM.

## Useful utilities

- **Specify known zero effects.** Append `--exclude-effect Al Cr` to a training command to fix only the Al–Cr pairwise term to zero. Neither input variable is removed, and the Al–Cr–Mo term is not excluded automatically. Repeat the flag for each additional effect, for example `--exclude-effect Mo --exclude-effect Al Cr`. The checkpoint preserves these exclusions for prediction and export; `effect_summary.csv` marks them with `prior_excluded` to distinguish them from effects set to zero by entmax.
- **Run ablations.** Use `--ablation no_purity`, `no_sparse_loss`, or `mse_only` to disable regularizers. `no_dense` allocates all AdamW epochs to sparse training, whereas `no_sparse_stage` allocates them all to dense training and retains the equal gates. Both keep the specified total number of epochs and the L-BFGS refinement setting.
- **Export pairwise Excel tables.** `export_effects_xlsx.py` is called by `export_effects.py` and should stay beside it. It is not a separate analysis step.

## An example workflow for using GAHGNN

The nonlinear analytical function (NAF) example uses three variables and provides fixed training and validation CSVs. No residual-generation or grid-generation scripts are needed.

1. **Train a single model for all three orders:**

   ```sh
   python main.py --train_data ./data/NAF/train_set.csv --val_data ./data/NAF/val_set.csv --embedding ./data/NAF/onehot-embedding.json --savepath ./runs/NAF_seed11 --size 3 --seed 11
   ```

2. **Predict the validation samples and obtain their native effects:**

   ```sh
   python predict.py --sample ./data/NAF/val_set.csv --modelpath ./runs/NAF_seed11 --savepath ./runs/NAF_seed11/validation
   ```

3. **Export the learned effects on a regular grid:**

   ```sh
   python export_effects.py --run ./runs/NAF_seed11 --points 51 --output ./runs/NAF_seed11/effects
   ```

To skip training and use the included NAF checkpoint, replace `./runs/NAF_seed11` in `--modelpath` or `--run` with `./data/NAF/best.pth.tar`. To set all NAF export axes to $[-\pi/2,\pi/2]$ explicitly, add `--lower -1.5707963267948966 --upper 1.5707963267948966`.

## Example datasets and checkpoints

| Case | Directory | Included files |
| --- | --- | --- |
| Nonlinear analytical function (NAF) | `data/NAF/` | `train_set.csv`, `val_set.csv`, `onehot-embedding.json`, and the `size=3`, seed 11 checkpoint `best.pth.tar`. |
| Landau ($r_\phi=+1$) | `data/Landau_r1/` | `train_set.csv`, `val_set.csv`, and `embedding.json`. |
| Landau ($r_\phi=-1$) | `data/Landau_rminus1/` | `train_set.csv`, `val_set.csv`, and `embedding.json`. |
| High-entropy alloy catalysis (HEAC) | `data/HEAC/` | `train_set.csv`, `val_set.csv`, and `embedding.json`. See the encoding note below. |
| Nb alloys | `data/Nb/` | Five `size=1` composition-only checkpoints, `seed1best.pth.tar` through `seed5best.pth.tar`, for seeds 11, 23, 37, 53, and 71, respectively. The training and validation CSVs are not included in this directory. |

**HEAC element encoding.** The HEAC CSVs use integer identifiers `26`, `27`, `28`, `29`, and `30` for Fe, Co, Ni, Cu, and Zn. To use the element-one-hot model, use the matching HEAC code package: it converts each identifier to a five-component element vector inside the model, without changing the CSV. The root `predict.py` uses scalar inputs and cannot load a HEAC element-one-hot checkpoint. Use the HEAC package's `predict.py` for those checkpoints and evaluate effects on the five discrete values, not on a continuous grid.

**Nb alloy checkpoints.** Pass an individual checkpoint file directly to `--modelpath` or `--run`. For summaries across all five models, export each model on the same grid and aggregate the curves point by point; a single checkpoint does not reproduce the mean curve or its seed-to-seed variation.

## Acknowledgement

GAHGNN builds on components of [Roost](https://github.com/CompRhys/roost), distributed under the MIT License. Please also cite the [Roost paper](https://doi.org/10.1038/s41467-020-19964-7) when appropriate.
