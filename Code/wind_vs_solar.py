import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from optimiser import optimise_bess
from profile import generate_hourly_historical_solar_profile, parse_renewables_ninja
from data_prep.reader import get_val
from lcoe_helpers import calculate_solar_bess_lcoe, calculate_wind_bess_lcoe

CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")

# lat and long for solar
latitude = 51.370
longitude = -1.846

solar_profile = generate_hourly_historical_solar_profile(latitude, longitude, solar_year=2023)
wind_profile = parse_renewables_ninja(os.path.join(INPUT_PATH, "gb_wind_52.7510_2.1016_2023.csv"))

capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))

print(wind_profile.max())

AVAILABILITIES = [round(a, 2) for a in np.arange(0.05, 1.001, 0.05)]
country = "United Kingdom"
BASE_YEAR = 2024
years = [2024, 2030]

solar_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "solar")
wind_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "offshore wind")
bess_energy_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex_e", "BESS")
bess_power_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex_p", "BESS")

all_results = []
breakdown_rows = []
rows_displayed = 0

def display_row(tech, avail, ren_cap, bess_energy, lcoe):
    """Pretty-print a single summary row."""
    print(f"{tech:10s} | AF={avail:4.2f} | "
          f"Renewable={ren_cap:6.2f} MW | "
          f"BESS={bess_energy:6.2f} MWh | "
          f"LCOE={lcoe:7.2f}")

# =========================================
# WIND OPTIMISATION LOOP
# =========================================
print("\n=== Running Wind+BESS optimisation ===")

for avail in tqdm(AVAILABILITIES, desc=f"{country} Wind", leave=True):

    cost, wind_cap, bess_energy, bess_power, results_data, cycles = optimise_bess(
        wind_profile,
        wind_capex_base,
        bess_energy_capex_base,
        bess_power_capex_base,
        availability=avail
    )

    rid = len(all_results)
    for year in years:
        sb = calculate_wind_bess_lcoe(
            country=country,
            year=year,
            wind_capacity_mw=wind_cap,
            bess_energy_mwh=bess_energy,
            bess_power_mw=bess_power,
            availability=avail,
            bess_cycles=cycles,
            capex_opex_df=capex_opex_df,
            discount_rate=0.08,
            lifetime=25,
            result_id=rid
        )

        all_results.append({
            "Result_ID": rid,
            "Country": country,
            "Year": year,
            "Tech": "Wind+BESS",
            "Availability": avail,
            "LCOE": round(sb["LCOE"], 2),
            "Total_Capex": sb["Total_Capex"],
            "Renewable_Capacity_MW": wind_cap,
            "BESS_Energy_MWh": bess_energy,
            "BESS_Power_MW": bess_power,
        })

        # ---- Add breakdown rows ----
        for comp, row_b in sb["Breakdown"].iterrows():
            breakdown_rows.append({
                "Result_ID": rid,
                "Country": country,
                "Year": year,
                "Tech": "Wind+BESS",
                "Availability": avail,
                "Component": comp,
                "Value": float(row_b["Value"]),
            })

    # ---- live print ----
    display_row("Wind+BESS", avail, wind_cap, bess_energy, round(sb["LCOE"], 2))

# =========================================
# SOLAR OPTIMISATION LOOP
# =========================================
print("\n=== Running Solar+BESS optimisation ===")

for avail in tqdm(AVAILABILITIES, desc=f"{country} Solar", leave=True):

    cost, solar_cap, bess_energy, bess_power, results_data, cycles = optimise_bess(
        solar_profile,
        solar_capex_base,
        bess_energy_capex_base,
        bess_power_capex_base,
        availability=avail
    )

    rid = len(all_results)

    for year in years:
        sb = calculate_solar_bess_lcoe(
            country=country,
            year=year,
            solar_capacity_mw=solar_cap,
            bess_energy_mwh=bess_energy,
            bess_power_mw=bess_power,
            availability=avail,
            bess_cycles=cycles,
            capex_opex_df=capex_opex_df,
            discount_rate=0.08,
            lifetime=25,
            result_id=rid
        )

        all_results.append({
            "Result_ID": rid,
            "Country": country,
            "Year": year,
            "Tech": "Solar+BESS",
            "Availability": avail,
            "LCOE": round(sb["LCOE"], 2),
            "Total_Capex": sb["Total_Capex"],
            "Renewable_Capacity_MW": solar_cap,
            "BESS_Energy_MWh": bess_energy,
            "BESS_Power_MW": bess_power,
        })

        for comp, row_b in sb["Breakdown"].iterrows():
            breakdown_rows.append({
                "Result_ID": rid,
                "Country": country,
                "Year": year,
                "Tech": "Solar+BESS",
                "Availability": avail,
                "Component": comp,
                "Value": float(row_b["Value"]),
            })

    # ---- live print ----
    display_row("Solar+BESS", avail, solar_cap, bess_energy, round(sb["LCOE"], 2))

# =========================================
# EXPORT RESULTS
# =========================================

results_df = pd.DataFrame(all_results)
breakdown_df = pd.DataFrame(breakdown_rows)

results_df.to_csv(OUTPUT_PATH + r"\renewable_bess_results.csv", index=False)
breakdown_df.to_csv(OUTPUT_PATH + r"\renewable_bess_breakdown.csv", index=False)

print("Saved results:")
print(results_df.head())
print("\nSaved breakdown:")
print(breakdown_df.head())