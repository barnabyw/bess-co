import os

base_path = r"C:\Users\barnaby.winser\Documents\solar bes"
input_path = os.path.join(base_path, "input data")
output_path = os.path.join(base_path, "output")

solar_cost_per_mw = 400
bess_power_cost_per_mw = 0
bess_energy_cost_per_mwh = 160
efficiency = 0.9
M = 1e5  # Big-M for binary control
start_soc = 0.5
target = 0.95
load = 100