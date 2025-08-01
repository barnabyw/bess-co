import os
import pandas as pd

base_path = r"C:\Users\barna\OneDrive\Documents\Solar_BESS"
input_path = os.path.join(base_path)
output_path = os.path.join(base_path, "output")

# Years
years = list(range(2020, 2031))
start_year = years[0]

# Initial CAPEX values ($k/unit)
solar_cost_per_mw_2020 = 400            # $k/MW
bess_power_cost_per_mw_2020 = 200       # Example
bess_energy_cost_per_mwh_2020 = 160     # $k/MWh

# Annual reduction rates (learning curves)
solar_reduction_rate = 0.05             # 5% per year
bess_power_reduction_rate = 0.07        # 7%
bess_energy_reduction_rate = 0.09       # 9%

efficiency = 0.9
M = 1e5  # Big-M for binary control
start_soc = 0.5
target = 0.95
load = 100

capex_learning_df = pd.DataFrame({"year": years})

capex_learning_df["solar_cost_per_mw"] = solar_cost_per_mw_2020 * (
    (1 - solar_reduction_rate) ** (capex_learning_df["year"] - start_year)
)
capex_learning_df["bess_power_cost_per_mw"] = bess_power_cost_per_mw_2020 * (
    (1 - bess_power_reduction_rate) ** (capex_learning_df["year"] - start_year)
)
capex_learning_df["bess_energy_cost_per_mwh"] = bess_energy_cost_per_mwh_2020 * (
    (1 - bess_energy_reduction_rate) ** (capex_learning_df["year"] - start_year)
)

