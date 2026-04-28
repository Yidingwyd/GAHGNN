# -*- coding: utf-8 -*-
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def dgsm(x, y):
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    x_range = x.max() - x.min()
    if x_range == 0:
        return np.nan
    return y.diff().abs().iloc[1:].sum() / x_range


input_file = Path("main_effect.xlsx")
output_file = Path("main_effect_result.xlsx")

sheet1 = pd.read_excel(input_file, sheet_name=0)
sheet2 = pd.read_excel(input_file, sheet_name=1, header=None)

if "composition" not in sheet1.columns:
    raise KeyError("Sheet1 must contain a 'composition' column")

composition_dicts = sheet1["composition"].apply(
    lambda v: ast.literal_eval(v) if isinstance(v, str) else v
)

elements = list(composition_dicts.iloc[0].keys())

if sheet2.shape[1] != len(elements):
    raise ValueError(
        f"Sheet2 has {sheet2.shape[1]} columns, but composition has {len(elements)} elements"
    )

if sheet1.shape[0] != sheet2.shape[0]:
    raise ValueError(
        f"Sheet1 has {sheet1.shape[0]} rows, but Sheet2 has {sheet2.shape[0]} rows"
    )

wide_pairs = pd.DataFrame({"row_index": sheet1.index})

if "Unnamed: 0" in sheet1.columns:
    wide_pairs["sample_id"] = sheet1["Unnamed: 0"]

for j, element in enumerate(elements):
    wide_pairs[f"{element}_composition"] = composition_dicts.apply(lambda d: d[element])
    wide_pairs[f"{element}_main_effect"] = sheet2.iloc[:, j]

summary_rows = []
for j, element in enumerate(elements):
    x = wide_pairs[f"{element}_composition"]
    y = wide_pairs[f"{element}_main_effect"]
    summary_rows.append(
        {
            "element_order": j,
            "element": element,
            "n": len(x),
            "composition_min": x.min(),
            "composition_max": x.max(),
            "composition_mean": x.mean(),
            "composition_var": x.var(),
            "main_effect_min": y.min(),
            "main_effect_max": y.max(),
            "main_effect_mean": y.mean(),
            "main_effect_var": y.var(),
            "dgsm": dgsm(x, y),
        }
    )

summary = pd.DataFrame(summary_rows)

with pd.ExcelWriter(output_file) as writer:
    wide_pairs.to_excel(writer, sheet_name="wide_pairs", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print(f"Saved: {output_file}")
print("Sheets: wide_pairs, summary")
