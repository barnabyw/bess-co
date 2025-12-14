import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. File path to your Excel file
# ------------------------------------------------------------
excel_file_path = r'C:\Users\barna\OneDrive\Documents\Solar_BESS\inputs\raw\IRENA summary.xlsx'

# ------------------------------------------------------------
# 2. Input and output sheet names
# ------------------------------------------------------------
input_sheet_name = 'Original Data'
output_sheet_name = 'Filled_Interpolated_Data2'

# ------------------------------------------------------------
# Load the input sheet
# ------------------------------------------------------------
df = pd.read_excel(excel_file_path, sheet_name=input_sheet_name, index_col=0)
df.index = df.index.astype(int)   # Ensure year index is integer

# Identify the World column
WORLD_COL = "World"

if WORLD_COL not in df.columns:
    raise ValueError("Your dataset must contain a column named 'World'.")

world = df[WORLD_COL]


# ------------------------------------------------------------
# Refined Ratio-to-World Filling Function
# ------------------------------------------------------------
def fill_column_ratio_to_world(col_series, world_series):
    """
    Fills missing values using:
    - calculated ratio where available
    - linear interpolation between ratio anchors
    - backward fill using earliest ratio
    - forward fill using latest ratio
    """

    # Compute ratio where possible
    ratio = col_series / world_series
    ratio_valid = ratio.dropna()

    if ratio_valid.empty:
        return col_series  # No anchors at all

    # Reindex to full year range and interpolate BETWEEN anchors
    ratio_full = ratio_valid.reindex(world_series.index).interpolate(method='linear')

    # NEW: backward fill using earliest ratio
    ratio_full = ratio_full.bfill()

    # NEW: forward fill using latest ratio
    ratio_full = ratio_full.ffill()

    # Compute filled values
    filled = world_series * ratio_full

    # Preserve original non-missing data
    filled[col_series.notna()] = col_series[col_series.notna()]

    return round(filled,0)

# ------------------------------------------------------------
# Apply the method to every country column
# ------------------------------------------------------------
filled_df = df.copy()

for col in df.columns:
    if col == WORLD_COL:
        continue
    filled_df[col] = fill_column_ratio_to_world(df[col], world)


# ------------------------------------------------------------
# Write output back to a new sheet in the same Excel file
# ------------------------------------------------------------
with pd.ExcelWriter(excel_file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    filled_df.to_excel(writer, sheet_name=output_sheet_name)

print("✔ Gap filling complete.")
print(f"✔ Results written to sheet: '{output_sheet_name}'")
