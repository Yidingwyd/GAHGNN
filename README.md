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

As an example, one can refer to the [EAF dataset](https://github.com/Yidingwyd/GAHGNN/blob/main/datasets/EAF/eaf.csv), of which the data are generated from an analytical nonlinear function with up to 3-order interactions.

The **input dataset** of GAHGNN should be saved in a `.csv` file:

- First column: `material_id`, ID of each sample.
- Second column: `composition`, nodes in the sample, represented as a Python dictionary:
  - Keys: name of each node (variables);
  - Values: weight of each node.
- Third column: `target`, target of the sample.

The **node features** of GAHGNN should be saved as a python dictionary in a `.json` file:

- Keys: name of the variables, which should be the same with that in the **input dataset**;
- Values: features to describe the variables. You can either use one-hot embeddings or physical descriptors.

Based on the input files, the program will automatically generate PyTorch tensors in a style of hypergraph representation. 

## Train a GAHGNN model
You can train a GAHGNN model by:  
```
python main.py --train_data train_set_path --val_data val_set_path --embedding node_feature_path --savepath model_save_path --size size_of_the_edge_in_the_model
```

For example:
```
python main.py --train_data ./datasets/eaf/train_set.csv --val_data ./datasets/eaf/val_set.csv --embedding ./datasets/eaf/onehot-embedding.json --savepath ./datasets/eaf/0/ --size 1
```
By running the above command, you can train a 1-uniform GAHGNN that contains only self-loops. In staged training, if you want to train other uniform GAHGNN variants, such as models containing only pairwise edges or 3-order hyperedges, `train_data` and `val_data` must be the residualized datasets, and the `size` parameter must be changed correspondingly. The detailed procedure is described later.

The input parameters of the model are summarized in the following table：  
![Table 1](https://github.com/Yidingwyd/NCGNN/blob/main/table1.png)  

## Interpretable analysis and prediction using a trained GAHGNN model
You do these by:
```
python predict.py --sample dataset_path --embedding node_feature_path --modelpath model_path --savepath results_save_path.xlsx
```
The `sample` parameter can be the data you want to predict, or the data on which you want to perform interpretability analysis.

The results are saved as an .xlsx file.

In Sheet1, the output column contains the model’s predictions for different samples. 

Sheet2 contains the effect values for each data point. If a size-1 model is used, meaning the model contains only self-loops, these columns correspond to the main effects and are arranged in the order of the variables. If a size-2 model is used, meaning the model contains only ordinary edges, these columns correspond to pairwise interactions and are arranged according to the combination order of the variables. The same rule applies to higher-order models.

## Useful utils
- [get_GAHGNN_data.py](https://github.com/Yidingwyd/GAHGNN/blob/main/utils/get_GAHGNN_data.py): If your dataset is stored in an Excel file, with the first n-1 columns representing different variable values and the last column representing the target, you can use this Python script to generate a CSV file compatible with GAHGNN.
- [res_cal.py](https://github.com/Yidingwyd/GAHGNN/blob/main/utils/res_cal.py): helps to compute the residuals between the predictions of lower-order models and the target values, and automatically saves them as CSV files in the format required for GAHGNN training.
- [get_main_data.py](https://github.com/Yidingwyd/GAHGNN/blob/main/utils/get_main_data.py) and [get_inter_data.py](https://github.com/Yidingwyd/GAHGNN/blob/main/utils/get_inter_data.py): When analyzing variable effects, it is usually necessary to examine each variable over its full value range. These two Python scripts can help generate main effect and pairwise interaction CSV files for GAHGNN prediction. In addition, [get_inter_data.py](https://github.com/Yidingwyd/GAHGNN/blob/main/utils/get_inter_data.py) can print the command that needs to be executed. You should modify num and elem_dict in the scripts according to your actual needs, as well as the print format in the last line of [get_inter_data.py](https://github.com/Yidingwyd/GAHGNN/blob/main/utils/get_inter_data.py).
- [analyze_main_effect.py](https://github.com/Yidingwyd/GAHGNN/blob/main/utils/analyze_main_effect.py) and [analyze_inter_effect.py](https://github.com/Yidingwyd/GAHGNN/blob/main/utils/analyze_inter_effect.py): These two scripts can help analyze the main effects and pairwise interaction effects predicted by GAHGNN. Note that `elem_dict` in `analyze_inter_effect.py` should be modified as needed to ensure that it is consistent with the one in `get_inter_data.py`.

## An example workflow for using GAHGNN
1. Train the 1-uniform GAHGNN
```
python main.py --train_data ./example/train_set.csv --val_data ./example/val_set.csv --embedding ./example/onehot-embedding.json --savepath ./example/0/ --size 1
```
2. Predict using the trained 1-uniform GAHGNN
```
python predict.py --sample ./example/train_set.csv --embedding ./example/onehot-embedding.json --modelpath ./example/0/best.pth.tar --savepath ./example/0/predict_train.xlsx
python predict.py --sample ./example/val_set.csv --embedding ./example/onehot-embedding.json --modelpath ./example/0/best.pth.tar --savepath ./example/0/predict_val.xlsx
```
3. Calculate the residual error

Run `res_cal.py` in `./example/0/`

4. Train the 2-uniform GAHGNN
```
python main.py --train_data ./example/0/train_res.csv --val_data ./example/0/val_res.csv --embedding ./example/onehot-embedding.json --savepath ./example/0-0/ --size 2
```
5. Predict using the trained 2-uniform GAHGNN
```
python predict.py --sample ./example/0/train_res.csv --embedding ./example/onehot-embedding.json --modelpath ./example/0-0/best.pth.tar --savepath ./example/0-0/predict_train.xlsx
python predict.py --sample ./example/0/val_res.csv --embedding ./example/onehot-embedding.json --modelpath ./example/0-0/best.pth.tar --savepath ./example/0-0/predict_val.xlsx
```
6. Calculate the residual error

Run `res_cal.py` in `./example/0-0/`

7. Train the 3-uniform GAHGNN
```
python main.py --train_data ./example/0-0/train_res.csv --val_data ./example/0-0/val_res.csv --embedding ./example/onehot-embedding.json --savepath ./example/0-0-0/ --size 3
```
8. Analyze the main effect

Run `get_main_data.py` in `./example/`

Then run
```
python predict.py --sample ./example/main.csv --embedding ./example/onehot-embedding.json --modelpath ./example/0/best.pth.tar --savepath ./example/0/main_effect.xlsx
```

Next run `analyze_main_effect.py` in `./example/0`

9. Analyze the pairwise interaction effect

Run `get_inter_data.py` in `./example/`, and the commands that are need to run next is printed.

Then run the printed commands.

Next run `analyze_inter_effect.py` in `./example/0-0`

10. Analyze the 3-order interaction effect
```
python predict.py --sample ./example/0-0-0/x1x2x3.csv --embedding ./example/onehot-embedding.json --modelpath ./example/0-0-0/best.pth.tar --savepath ./example/0-0-0/high_inter_effect.xlsx
```



