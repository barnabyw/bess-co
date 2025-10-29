# main_workflow.py
import pandas as pd
import os
from tqdm import tqdm  # A library for nice progress bars

# --- Import your custom modules ---
from reader import get_val
from profile import generate_hourly_solar_profile
from optimiser import optimise_bess
from lcoe_helpers import calculate_solar_bess_lcoe, calculate_conventional_lcoe
from lcoe.lcoe import lcoe

# --- Configuration ---
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

BASE_YEAR = 2024
YEARS = list(range(2010, 2025))
CONVENTIONAL_TECHS = ["Coal", "Gas"]

# Availability used by your simple Solar+BESS LCOE (per your helper)
availability = 0.8

# --- Load Data ---
print("Loading input data...")
countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted_2025USD.xlsx"))
print("Data loaded successfully.")

# --- Optional: specify which countries to run ---
target_countries = ["Chile", "Australia", "Spain"]  # or [] to process all
if target_countries:
    countries_to_process = countries_df[countries_df["Country"].isin(target_countries)]
    print(f"Running analysis for {len(countries_to_process)} selected countries: {', '.join(target_countries)}")
else:
    countries_to_process = countries_df
    print(f"Running analysis for all {len(countries_to_process)} countries.")

# --- Main Analysis Loop ---
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

    # Generate solar profile once per country (kept as-is for your optimiser)
    yearly_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)

    # --- Step 1: Optimize Solar+BESS capacity for the base year ---
    print(f"  Optimizing Solar+BESS for base year {BASE_YEAR}...")
    try:
        solar_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "Solar")
        bess_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "BESS")

        cost, solar_cap, bess_energy, results_1 = optimise_bess(
            yearly_profile, solar_capex_base, bess_capex_base
        )

        # FIX: pass `availability` (NOT the profile) to the helper
        result = calculate_solar_bess_lcoe(
            country, BASE_YEAR, solar_cap, bess_energy, availability, capex_opex_df
        )

        # FIX: result is a dict; store the LCOE value
        all_results.append({
            "Country": country, "Year": BASE_YEAR, "Tech": "Solar+BESS",
            "LCOE": result.get("LCOE") if result else None,
            "Cost": result.get("Total_Capex") if result else None,
            "Solar_Capacity_MW": solar_cap, "BESS_Energy_MWh": bess_energy,
        })
        print(f"  -> Optimal capacity for {country}: Solar={solar_cap:.2f} MW, BESS={bess_energy:.2f} MWh")

    except ValueError as e:
        print(f"  ERROR: Could not optimize for {country} in {BASE_YEAR}. Skipping. Reason: {e}")
        continue  # Skip to the next country if optimization fails

    # --- Step 2: Historical Solar+BESS LCOE with fixed capacities ---
    print(f"  Calculating historical Solar+BESS LCOE...")
    for year in YEARS:
        if year == BASE_YEAR:
            continue  # already done

        # FIX: pass `availability` (simple helper expects this)
        result = calculate_solar_bess_lcoe(
            country, year, solar_cap, bess_energy, availability, capex_opex_df
        )

        if result:
            all_results.append({
                "Country": country, "Year": year, "Tech": "Solar+BESS",
                "LCOE": result.get("LCOE"), "Cost": result.get("Total_Capex"),
                "Solar_Capacity_MW": solar_cap, "BESS_Energy_MWh": bess_energy,
            })

    # --- Step 3: Conventional tech LCOE across all years ---
    # The helper needs capacity_mw and capacity_factor.
    # We use 1.0 MW (scale-invariant) and read CF from the sheet per (country, year, tech).
    for tech in CONVENTIONAL_TECHS:
        print(f"  Calculating {tech} LCOE for all years...")
        for year in YEARS:
            try:
                cf = get_val(capex_opex_df, country, year, "capacity_factor", tech)
                result = calculate_conventional_lcoe(
                    country=country,
                    year=year,
                    tech=tech,
                    capacity_mw=1.0,               # FIX: explicit capacity for helper signature
                    capacity_factor=cf,             # FIX: pass CF from the table
                    capex_opex_df=capex_opex_df
                )
                if result:
                    all_results.append({
                        "Country": country, "Year": year, "Tech": tech,
                        "LCOE": result.get("LCOE"),
                        "Cost": result.get("Total_Capex"),
                        "Solar_Capacity_MW": None, "BESS_Energy_MWh": None,
                    })
            except ValueError as e:
                print(f"   - Skipping {tech} {year} for {country}: {e}")
                continue

# --- Finalize and Save Results ---
print("\nAnalysis complete. Compiling and saving results...")
results_df = pd.DataFrame(all_results)

# Reorder columns for clarity
output_cols = [
    "Country", "Year", "Tech", "LCOE", "Cost",
    "Solar_Capacity_MW", "BESS_Energy_MWh"
]
results_df = results_df[output_cols]

output_file = os.path.join(OUTPUT_PATH, "lcoe_results.csv")
results_df.to_csv(output_file, index=False)

print(f"Results successfully saved to {output_file}")
print(results_df.head())
