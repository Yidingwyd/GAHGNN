# GAHGNN
Generalized additive hypergraph neural network
# Brief review
GAHGNN is an intrinsically interpretable additive framework that represents variables as nodes, main effects as self-loops, pairwise interactions as edges, and higher-order interactions as hyperedges in a hypergraph. This design makes GAHGNN especially suitable for scientific applications where interpretability, interaction modeling, and structural transparency are important.
# How to cite
Our manuscript is submitted.
# Prerequisites
This Python package requires: pytorch, torch_scatter, scikit-learn, pandas, numpy, entmax, tqdm, json.
# Usage
## Input of GAHGNN
Two files are required to be input into GAHGNN:  
* Input dataset `.csv`.  
* Node features to describe the variables `.json`.
As an example, one can refer to the [EPA dataset](https://github.com/Yidingwyd/NCGNN/blob/main/Kfold/cpa/cpa_formation_energy_per_atom.json), of which the data are generated from an analytical nonlinear function with up to 3-order interactions.
The **input dataset** of GAHGNN should be saved in a `.csv` file:
-First column: 'material_id', ID of each sample;
-Second column: 'composition', nodes in the sample, represented as a python dictionary:
      -keys: name of each nodes;
      -values: weight of each nodes.
-Third column: 'target', target of the sample.


