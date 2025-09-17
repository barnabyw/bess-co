import os
import pandas as pd

base_path = r"C:\Users\barna\OneDrive\Documents\Solar_BESS"
input_path = os.path.join(base_path)
output_path = os.path.join(base_path, "output")

# Years
years = list(range(2020, 2031))
start_year = years[0]
efficiency = 0.95
M = 1e5  # Big-M for binary control
start_soc = 0.5
target = 0.8
load = 100

# Initial CAPEX values ($k/unit)
solar_cost_per_mw_2020 = 400            # $k/MW
bess_power_cost_per_mw_2020 = 0       # Example
bess_energy_cost_per_mwh_2020 = 160     # $k/MWh

solar_cost_per_mw = 400
bess_energy_cost_per_mwh = 160

# Annual reduction rates (learning curves)
solar_reduction_rate = 0.05             # 5% per year
bess_power_reduction_rate = 0.15        # 7%
bess_energy_reduction_rate = 0.09       # 9%

capex_learning_df = pd.read_excel(os.path.join(base_path,"learning curves\\capex.xlsx"), sheet_name="capex")

if __name__ == "__main__":
    print(capex_learning_df)