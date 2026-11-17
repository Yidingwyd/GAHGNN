
import numpy as np

try:
    import pandas as pd
    from openpyxl.styles import Alignment, Border, Font, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError as exc:
    raise ImportError(
        "Order-2 XLSX export requires pandas and openpyxl. "
        "Install them in the active Python environment: python -m pip install pandas openpyxl"
    ) from exc


def write_effect_xlsx(path, variables, axes, values, value_column):
    row_variable, column_variable = variables
    row_axis, column_axis = (np.asarray(axis, dtype=float) for axis in axes)
    values = np.asarray(values, dtype=float)
    nr, nc = len(row_axis), len(column_axis)
    if not nr or not nc or values.size != nr * nc:
        raise ValueError("Pair effect values must match the two nonempty axes.")
    if not all(np.isfinite(array).all() for array in (row_axis, column_axis, values)):
        raise ValueError("Pair effect coordinates and values must be finite.")
    if nr * nc + 1 > 1_048_576 or nc + 1 > 16_384:
        raise ValueError("Pair-effect grid exceeds Excel's row or column limits.")

    grid = values.reshape(nr, nc)
    long = pd.DataFrame({
        row_variable: np.repeat(row_axis, nc),
        column_variable: np.tile(column_axis, nr),
        value_column: grid.ravel(),
    })
    corner = f"{row_variable} \\ {column_variable}"
    matrix = pd.DataFrame(grid, index=row_axis, columns=column_axis)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        long.to_excel(writer, sheet_name="long", index=False)
        matrix.to_excel(writer, sheet_name="matrix", index_label=corner)
        for name in ("long", "matrix"):
            style_sheet(writer.sheets[name], is_matrix=(name == "matrix"))


def style_sheet(sheet, is_matrix):
    body_font = Font(name="Arial", size=11)
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFFFF")
    header_fill = PatternFill("solid", fgColor="FF203864")
    row_axis_fill = PatternFill("solid", fgColor="FFE8EDF5")
    header_alignment = Alignment(horizontal="center", vertical="center")
    no_border = Border()
    coord_format = "0.000000"
    effect_format = "0.000000000E+00"
    n_columns = sheet.max_column

    sheet.sheet_view.showGridLines = False
    sheet.sheet_format.defaultRowHeight = 18
    sheet.row_dimensions[1].height = 24
    sheet.freeze_panes = "B2" if is_matrix else "A2"
    for cell in sheet[1]:
        if isinstance(cell.value, str):
            cell.data_type = "s"
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = no_border
        if is_matrix and cell.column > 1:
            cell.number_format = coord_format
        label_length = len(cell.value) if isinstance(cell.value, str) else 0
        sheet.column_dimensions[get_column_letter(cell.column)].width = min(50, max(22, label_length + 3))

    for row in sheet.iter_rows(min_row=2, max_col=n_columns):
        for cell in row:
            is_coordinate = cell.column <= (1 if is_matrix else 2)
            cell.font = body_font
            cell.number_format = coord_format if is_coordinate else effect_format
            if is_matrix and is_coordinate:
                cell.fill = row_axis_fill
                cell.border = no_border
