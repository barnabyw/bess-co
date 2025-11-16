import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from profile import generate_hourly_solar_profile
from reader import get_val
from optimiser import optimise_bess  # Adjust path

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
long_rows = []   # list of dicts (fastest structure for large tables)

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

    # --- compute solar breakdown ---
    solar_gen = ts["Solar_Generation_MWh"]
    bess_flow = ts["BESS_Flow_MWh"]

    # where bess_flow < 0 → charging
    solar_to_bess = np.clip(-bess_flow, 0, None)

    # direct use = min(solar_gen - charging, demand - discharge)
    solar_used = np.minimum(solar_gen - solar_to_bess,
                            LOAD - np.clip(bess_flow, 0, None))

    solar_used = np.clip(solar_used, 0, None)

    # curtailment = remaining solar not used or stored
    solar_curtailed = solar_gen - solar_used - solar_to_bess
    solar_curtailed = np.clip(solar_curtailed, 0, None)

    bess_discharge = np.clip(bess_flow, 0, None)

    served = ts["Energy_Served_MWh"]
    unserved = LOAD - served
    unserved = np.clip(unserved, 0, None)

    # --------------------------------------------
    # LONG FORMAT STACKED ROWS
    # --------------------------------------------
    hours = ts["Hour"]

    variables = {
        "solar_used": solar_used,
        "solar_to_bess": solar_to_bess,
        "solar_curtailed": solar_curtailed,
        "bess_discharge": bess_discharge,
        "soc": ts["SOC_MWh"],
        "energy_served": served,
        "energy_unserved": unserved,
    }

    for var_name, series in variables.items():
        for h, v in zip(hours, series):
            long_rows.append({
                "Hour": h,
                "Variable": var_name,
                "Value": v,
                "Availability": avail,
                "Country": COUNTRY
            })

# ---------------------------------------------
# EXPORT
# ---------------------------------------------
final_df = pd.DataFrame(long_rows)

outfile = os.path.join(
    OUTPUT_PATH,
    f"long_timeseries_{COUNTRY.replace(' ', '_')}.csv"
)

final_df.to_csv(outfile, index=False)

print(f"\nSaved BI-ready long dataframe ({len(final_df)} rows) to:")
print(outfile)
