# main_workflow.py
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# === Import custom modules ===
from Code.data_prep.reader import get_val
from profile import generate_hourly_solar_profile
from optimiser import optimise_bess
from lcoe_helpers import calculate_solar_bess_lcoe, calculate_conventional_lcoe

# === Configuration ===
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)
output_file = os.path.join(OUTPUT_PATH, "lcoe_results.csv")

BASE_YEAR = 2024
YEARS = list(range(2015, 2025))
CONVENTIONAL_TECHS = ["Coal", "Gas"]

# === Sensitivity Cases ===
SENSITIVITY_CASES = [
    {"name": "Default", "discount_rate": None, "lifetime": None},
    {"name": "DR=0.08_LT=25", "discount_rate": 0.08, "lifetime": 25},
]

# === Load Data ===
countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))

# === Select Countries ===
target_countries = ["United States", "Australia", "Spain", "United Kingdom", "Saudi Arabia", "China"]
if target_countries:
    countries_to_process = countries_df[countries_df["Country"].isin(target_countries)]
else:
    countries_to_process = countries_df

# === Availability Sweep ===
AVAILABILITIES = [round(a, 2) for a in np.arange(0.05, 1.001, 0.05)]

# === Storage ===
all_results = []

# === Overall Progress Bar ===
overall_pbar = tqdm(
    total=len(countries_to_process),
    desc="Overall Progress",
    colour="green"
)

for _, row in countries_to_process.iterrows():

    country = row["Country"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    # Generate solar profile once per country
    yearly_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)

    # Base-year CAPEX
    try:
        solar_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "solar")
        bess_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "bess")
    except ValueError:
        overall_pbar.update(1)
        continue

    # Availability Sweep
    for avail in tqdm(AVAILABILITIES, desc=f"Avail Sweep ({country})", leave=False):

        # --- Step 1: Optimise Solar+BESS ---
        try:
            cost, solar_cap, bess_energy, _ = optimise_bess(
                yearly_profile, solar_capex_base, bess_capex_base, availability=avail
            )
        except ValueError:
            continue

        # --- Step 2: Solar+BESS LCOE (Base Year + Sensitivities) ---
        for sens in SENSITIVITY_CASES:

            rid = len(all_results)

            sb_result = calculate_solar_bess_lcoe(
                country=country,
                year=BASE_YEAR,
                solar_capacity_mw=solar_cap,
                bess_capacity_mwh=bess_energy,
                availability=avail,
                capex_opex_df=capex_opex_df,
                result_id=rid,
                discount_rate=sens["discount_rate"],
                lifetime=sens["lifetime"],
            )

            if sb_result:
                all_results.append({
                    "Result_ID": rid,
                    "Sensitivity": sens["name"],
                    "Country": country, "Year": BASE_YEAR, "Tech": "Solar+BESS",
                    "Availability": avail,
                    "LCOE": round(sb_result["LCOE"], 2),
                    "Cost": sb_result["Total_Capex"],
                    "Solar_Capacity_MW": round(solar_cap, 1),
                    "BESS_Energy_MWh": round(bess_energy, 1),
                })

        # --- Step 3: Historical Solar+BESS LCOE ---
        for year in YEARS:
            if year == BASE_YEAR:
                continue

            for sens in SENSITIVITY_CASES:

                rid = len(all_results)

                hist_result = calculate_solar_bess_lcoe(
                    country=country,
                    year=year,
                    solar_capacity_mw=solar_cap,
                    bess_capacity_mwh=bess_energy,
                    availability=avail,
                    capex_opex_df=capex_opex_df,
                    result_id=rid,
                    discount_rate=sens["discount_rate"],
                    lifetime=sens["lifetime"],
                )

                if hist_result:
                    all_results.append({
                        "Result_ID": rid,
                        "Sensitivity": sens["name"],
                        "Country": country, "Year": year, "Tech": "Solar+BESS",
                        "Availability": avail,
                        "LCOE": round(hist_result["LCOE"], 2),
                        "Cost": hist_result["Total_Capex"],
                        "Solar_Capacity_MW": round(solar_cap, 1),
                        "BESS_Energy_MWh": round(bess_energy, 1),
                    })

        # --- Step 4: Conventional Techs ---
        for tech in CONVENTIONAL_TECHS:
            for year in YEARS:
                for sens in SENSITIVITY_CASES:

                    rid = len(all_results)

                    conv_result = calculate_conventional_lcoe(
                        country=country,
                        year=year,
                        tech=tech,
                        capacity_mw=1.0,
                        capacity_factor=avail,
                        capex_opex_df=capex_opex_df,
                        result_id=rid,
                        discount_rate=sens["discount_rate"],
                        lifetime=sens["lifetime"],
                    )

                    if conv_result:
                        all_results.append({
                            "Result_ID": rid,
                            "Sensitivity": sens["name"],
                            "Country": country, "Year": year, "Tech": tech,
                            "Availability": avail,
                            "LCOE": conv_result["LCOE"],
                            "Cost": conv_result["Total_Capex"],
                            "Solar_Capacity_MW": None,
                            "BESS_Energy_MWh": None,
                        })

    overall_pbar.update(1)

overall_pbar.close()

# === Save Output ===
results_df = pd.DataFrame(all_results)

output_cols = [
    "Result_ID",
    "Sensitivity",
    "Country", "Year", "Tech", "Availability",
    "LCOE", "Cost",
    "Solar_Capacity_MW", "BESS_Energy_MWh",
]

results_df = results_df[output_cols]
results_df.to_csv(output_file, index=False)

print("Done! Results saved to", output_file)
