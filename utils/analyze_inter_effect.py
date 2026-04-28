# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from itertools import combinations, product


elem_dict = {
    'one': [26, 27, 28, 29, 30],
    'two': [26, 27, 28, 29, 30],
    'three': [26, 27, 28, 29, 30],
    'four': [26, 27, 28, 29, 30],
    'five': [26, 27, 28, 29, 30],
    'six': [26, 27, 28, 29, 30],
    'seven': [26, 27, 28, 29, 30],
    'eight': [26, 27, 28, 29, 30],
    'nine': [26, 27, 28, 29, 30],
    'ten': [26, 27, 28, 29, 30],
    'eleven': [26, 27, 28, 29, 30],
    'twelve': [26, 27, 28, 29, 30],
    'thirteen': [26, 27, 28, 29, 30],
    'fourteen': [26, 27, 28, 29, 30],
    'fifteen': [26, 27, 28, 29, 30],
    'sixteen': [26, 27, 28, 29, 30],
}

input_dir = '.'
output_file = 'inter_effect_result.xlsx'

def make_sheet_name(name, used_names):
    sheet_name = re.sub(r'[:\\/?*\[\]]', '_', name)[:31]
    base = sheet_name
    index = 1
    while sheet_name in used_names:
        suffix = f'_{index}'
        sheet_name = f'{base[:31 - len(suffix)]}{suffix}'
        index += 1
    used_names.add(sheet_name)
    return sheet_name


def build_inter_effect_result(elem_dict, input_dir='.', output_file='inter_effect_result.xlsx'):
    used_sheet_names = set()
    pair_info = []

    with pd.ExcelWriter(output_file) as writer:
        for col_index, (x_name, y_name) in enumerate(combinations(elem_dict.keys(), 2)):
            x_values = list(elem_dict[x_name])
            y_values = list(elem_dict[y_name])
            xy_pairs = list(product(x_values, y_values))
            expected_rows = len(xy_pairs)

            input_file = os.path.join(input_dir, f'{x_name}{y_name}.xlsx')
            if not os.path.exists(input_file):
                raise FileNotFoundError(f'找不到文件：{input_file}')

            raw = pd.read_excel(input_file, sheet_name=1, header=None)

            if raw.shape[1] <= col_index:
                raise ValueError(
                    f'{input_file} 的列数为 {raw.shape[1]}，'
                    f'但当前组合 {x_name}-{y_name} 需要读取第 {col_index + 1} 列。'
                )

            z_values_raw = raw.iloc[:, col_index].dropna().reset_index(drop=True)

            if len(z_values_raw) != expected_rows:
                raise ValueError(
                    f'{input_file} 中第 {col_index + 1} 列有效行数为 {len(z_values_raw)}，'
                    f'但 {x_name}({len(x_values)}) × {y_name}({len(y_values)}) 应为 {expected_rows} 行。'
                )

            z_mean_raw = z_values_raw.mean()
            z_values = z_values_raw - z_mean_raw

            result = pd.DataFrame(xy_pairs, columns=[x_name, y_name])
            result['z'] = z_values
            result.insert(0, 'pair_order', col_index)
            result.insert(1, 'x_name', x_name)
            result.insert(2, 'y_name', y_name)

            sheet_name = make_sheet_name(f'{x_name}_{y_name}', used_sheet_names)
            result.to_excel(writer, sheet_name=sheet_name, index=False)

            pair_info.append(
                {
                    'pair_order': col_index,
                    'x_name': x_name,
                    'y_name': y_name,
                    'input_file': f'{x_name}{y_name}.xlsx',
                    'source_column_index_0_based': col_index,
                    'source_column_index_1_based': col_index + 1,
                    'x_count': len(x_values),
                    'y_count': len(y_values),
                    'xy_count': expected_rows,
                    'z_raw_mean_subtracted': z_mean_raw,
                    'z_sum_after_centering': z_values.sum(),
                    'z_var': z_values.var(),
                    'z_min': z_values.min(),
                    'z_max': z_values.max(),
                    'z_mean': z_values.mean(),
                    'sheet_name': sheet_name,
                }
            )

        pd.DataFrame(pair_info).to_excel(writer, sheet_name='summary', index=False)

    return output_file


if __name__ == '__main__':
    build_inter_effect_result(elem_dict, input_dir=input_dir, output_file=output_file)
    print(f'已保存：{output_file}')
