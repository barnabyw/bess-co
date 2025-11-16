import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from profile import generate_hourly_solar_profile
from reader import get_val
from optimiser import optimise_bess  # make sure this imports the new version

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))

COUNTRY = "Australia"
BASE_YEAR = 2024
LOAD = 1.0
AVAILABILITIES = [round(a, 2) for a in np.arange(0.05, 1.001, 0.05)]

# ---------------------------------------------
# FETCH LOCATION
# ---------------------------------------------
row = countries_df[countries_df["Country"] == COUNTRY]
if row.empty:
    raise ValueError(f"Country '{COUNTRY}' not found.")

lat = row.iloc[0]["Latitude"]
lon = row.iloc[0]["Longitude"]

# ---------------------------------------------
# SOLAR PROFILE
# ---------------------------------------------
solar_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)

# ---------------------------------------------
# COST INPUTS
# ---------------------------------------------
solar_capex = get_val(capex_opex_df, COUNTRY, BASE_YEAR, "capex", "solar")
bess_capex = get_val(capex_opex_df, COUNTRY, BASE_YEAR, "capex", "bess")

# ---------------------------------------------
# MAIN LOOP — BI-ready long format
# ---------------------------------------------
long_dfs = []   # list of DataFrames to concat

for avail in tqdm(AVAILABILITIES, desc="Availability Sweep"):

    cost, solar_cap, bess_energy, ts = optimise_bess(
        solar_profile,
        solar_capex,
        bess_capex,
        load=LOAD,
        availability=avail,
        return_timeseries=True
    )

    if ts is None:
        continue

    # Add metadata
    ts = ts.copy()
    ts["Availability"] = avail
    ts["Country"] = COUNTRY

    # Optional: keep raw solar generation as a separate variable too
    # Map model column names -> nice variable names for long format
    rename_map = {
        "Solar_Used_MWh":      "solar_used",
        "Solar_Charge_MWh":    "solar_to_bess",
        "Solar_Curtailed_MWh": "solar_curtailed",
        "BESS_Discharge_MWh":  "bess_discharge",
        "SOC_MWh":             "soc",
        "Energy_Served_MWh":   "energy_served",
        "Energy_Unserved_MWh": "energy_unserved",
        "Solar_Gen_MWh":       "solar_generation",  # if you want this too
    }

    # Only keep variables that actually exist in ts
    value_vars = [col for col in rename_map.keys() if col in ts.columns]
    ts = ts.rename(columns=rename_map)

    # Melt to long format
    long_ts = ts.melt(
        id_vars=["Hour", "Availability", "Country"],
        value_vars=[rename_map[c] for c in value_vars],
        var_name="Variable",
        value_name="Value",
    )

    long_dfs.append(long_ts)

# ---------------------------------------------
# EXPORT
# ---------------------------------------------
if long_dfs:
    final_df = pd.concat(long_dfs, ignore_index=True)

    outfile = os.path.join(
        OUTPUT_PATH,
        f"long_timeseries_{COUNTRY.replace(' ', '_')}.csv"
    )

    final_df.to_csv(outfile, index=False)

    print(f"\nSaved BI-ready long dataframe ({len(final_df)} rows) to:")
    print(outfile)
else:
    print("No results produced.")
