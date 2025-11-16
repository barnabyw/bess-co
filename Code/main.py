# main_workflow.py
import pandas as pd
import os
from tqdm import tqdm  # progress bars
import numpy as np

# === Import custom modules ===
from reader import get_val
from profile import generate_hourly_solar_profile
from optimiser import optimise_bess
from lcoe_helpers import calculate_solar_bess_lcoe, calculate_conventional_lcoe

# === Configuration ===
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

BASE_YEAR = 2024
YEARS = list(range(2015, 2025))
CONVENTIONAL_TECHS = ["Coal", "Gas"]

# Availability used by Solar+BESS LCOE
availability = 0.8

# === Load Data ===
print("Loading input data...")
countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))
print("Data loaded successfully.")

# === Optional: specify which countries to run ===
target_countries = ["United States", "Saudi Arabia", "Chile", "Australia", "Spain", "United Kingdom"]  # or [] to process all
if target_countries:
    countries_to_process = countries_df[countries_df["Country"].isin(target_countries)]
    print(f"Running analysis for {len(countries_to_process)} selected countries: {', '.join(target_countries)}")
else:
    countries_to_process = countries_df
    print(f"Running analysis for all {len(countries_to_process)} countries.")

# sweep 0.40..0.95 inclusive at 0.05 steps
AVAILABILITIES = [round(a, 2) for a in np.arange(0.05, 1.001, 0.05)]

# === Main Analysis Loop ===
all_results = []

for _, row in tqdm(
    countries_to_process.iterrows(),
    total=countries_to_process.shape[0],
    desc="Processing Countries"
):
    country = row["Country"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    print(f"\nProcessing {country}...")

    # Generate solar profile once per country (reused across availability sweeps)
    yearly_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)

    # Read base-year cost inputs once (shared across availability sweeps)
    try:
        solar_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "Solar")
        bess_capex_base  = get_val(capex_opex_df, country, BASE_YEAR, "capex", "BESS")
    except ValueError as e:
        print(f"  ERROR: Missing CAPEX for {country} {BASE_YEAR}: {e}")
        continue

    # === Availability sweep ===
    for avail in tqdm(AVAILABILITIES, desc=f"  Availability sweep", leave=False):
        # --- Step 1: Optimise Solar+BESS for this availability in base year ---
        try:
            cost, solar_cap, bess_energy, results_1 = optimise_bess(
                yearly_profile, solar_capex_base, bess_capex_base, availability=avail
            )

            sb_result = calculate_solar_bess_lcoe(
                country, BASE_YEAR, solar_cap, bess_energy, avail, capex_opex_df
            )

            all_results.append({
                "Country": country, "Year": BASE_YEAR, "Tech": "Solar+BESS",
                "Availability": avail,
                "LCOE": round(sb_result.get("LCOE"), 2),
                "Cost": sb_result.get("Total_Capex"),
                "Solar_Capacity_MW": round(solar_cap, 1),
                "BESS_Energy_MWh": round(bess_energy, 1),
            })
            print(f"  {country:<15} | Avail={avail:.2f} | LCOE={sb_result.get('LCOE'):.2f} | "
                  f"Solar={solar_cap:.1f} MW | BESS={bess_energy:.1f} MWh")

        except ValueError as e:
            print(f"   - Skipping Solar+BESS @ Avail {avail:.2f} for {country}: {e}")
            continue

        # --- Step 2: Historical Solar+BESS (fixed capacities, varying years) ---
        # (Your YEARS currently only includes 2024; keep structure for future years.)
        for year in YEARS:
            if year == BASE_YEAR:
                continue
            try:
                hist_result = calculate_solar_bess_lcoe(
                    country, year, solar_cap, bess_energy, avail, capex_opex_df
                )
                if hist_result:
                    all_results.append({
                        "Country": country, "Year": year, "Tech": "Solar+BESS",
                        "Availability": avail,
                        "LCOE": round(hist_result.get("LCOE"), 2),
                        "Cost": hist_result.get("Total_Capex"),
                        "Solar_Capacity_MW": round(solar_cap, 1),
                        "BESS_Energy_MWh": round(bess_energy, 1),
                    })
            except ValueError as e:
                print(f"   - Skipping Solar+BESS hist {year} @ Avail {avail:.2f}: {e}")

        # --- Step 3: Conventional techs for all years ---
        # Capacity factor should match the availability sweep value
        for tech in CONVENTIONAL_TECHS:
            for year in YEARS:
                try:
                    cf = avail  # match CF to the current availability
                    conv_result = calculate_conventional_lcoe(
                        country=country,
                        year=year,
                        tech=tech,
                        capacity_mw=1.0,
                        capacity_factor=cf,
                        capex_opex_df=capex_opex_df
                    )
                    if conv_result:
                        all_results.append({
                            "Country": country, "Year": year, "Tech": tech,
                            "Availability": avail,
                            "LCOE": conv_result.get("LCOE"),
                            "Cost": conv_result.get("Total_Capex"),
                            "Solar_Capacity_MW": None,
                            "BESS_Energy_MWh": None,
                        })
                except ValueError as e:
                    print(f"   - Skipping {tech} {year} @ Avail {avail:.2f} for {country}: {e}")
                    continue

# === Finalize and Save Results ===
print("\nAnalysis complete. Compiling and saving results...")
results_df = pd.DataFrame(all_results)

# Reorder columns for clarity (added 'Availability')
output_cols = [
    "Country", "Year", "Tech", "Availability", "LCOE", "Cost",
    "Solar_Capacity_MW", "BESS_Energy_MWh",
]
results_df = results_df[output_cols]

output_file = os.path.join(OUTPUT_PATH, "lcoe_results.csv")
results_df.to_csv(output_file, index=False)

print(f"Results successfully saved to {output_file}")
print(results_df.head())
