

import pandas as pd


def res_cal(input_file, output_file):
    df = pd.read_excel(input_file)
    
    new_df = df.iloc[:, :2].copy()
    new_df["target"] = df["target"] - df["output"]
    
    new_df.to_csv(output_file, index=False)



input_file = "./predict_train.xlsx"
output_file = "./train_res.csv"
res_cal(input_file, output_file)

input_file = "./predict_val.xlsx"
output_file = "./val_res.csv"
res_cal(input_file, output_file)

