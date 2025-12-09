import numpy as np
import pandas as pd
from pathlib import Path
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# === OUTPUT PATH (MAIN INPUT) ===
CWD = Path(__file__).resolve().parent
_DATA_DIR = os.path.join(CWD.parent.parent, "outputs")

# Load your data
df = pd.read_csv(_DATA_DIR + r"\lcoe_results - Copy.csv")

# Split the technologies
df = df[df["Country"] == "Australia"]
df = df[df["Year"] == 2024]
solar_df = df[df["Tech"] == "Solar+BESS"].sort_values("Availability")
gas_df   = df[df["Tech"] == "Coal"].sort_values("Availability")

# Interpolate LCOE as continuous functions
solar_lcoe = interp1d(solar_df["Availability"], solar_df["LCOE"], kind="cubic")
gas_lcoe   = interp1d(gas_df["Availability"],   gas_df["LCOE"],   kind="cubic")

# System cost function for availability 'a' of solar
def system_cost(a):
    gas_a = 1 - a
    # Enforce bounds to avoid interpolation outside domain
    if a < 0 or a > 1 or gas_a < 0 or gas_a > 1:
        return np.inf
    return a * solar_lcoe(a) + gas_a * gas_lcoe(gas_a)

# Minimise system cost
res = minimize_scalar(system_cost, bounds=(0,1), method="bounded")

opt_a = res.x
opt_cost = res.fun

print("Optimal solar+BESS availability:", opt_a)
print("Optimal gas availability:", 1 - opt_a)
print("Min system cost:", opt_cost)
print("Solar LCOE:", solar_lcoe(opt_a))
print("Gas LCOE:", gas_lcoe(1 - opt_a))
