# main_workflow.py
import pandas as pd
import os
from tqdm import tqdm  # A library for nice progress bars

# --- Import your custom modules ---
# Make sure these files are in the same directory or Python's path
from reader import get_val
from profile import generate_hourly_solar_profile
from optimiser import optimise_bess
from lcoe_helpers import calculate_solar_bess_lcoe, calculate_conventional_lcoe

# --- Configuration ---
# Use absolute paths or paths relative to the script location for robustness
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

BASE_YEAR = 2024
YEARS = list(range(2010, 2025))
CONVENTIONAL_TECHS = ["Coal", "Gas"]

# --- Load Data ---
print("Loading input data...")
countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted_2025USD.xlsx"))
print("Data loaded successfully.")

# --- Main Analysis Loop ---
all_results = []

# Use tqdm for a progress bar over the countries
for _, row in tqdm(countries_df.iterrows(), total=countries_df.shape[0], desc="Processing Countries"):
    country = row["Country"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    print(f"\nProcessing {country}...")

    # Generate solar profile once per country
    yearly_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)

    # --- Step 1: Optimize Solar+BESS capacity for the most recent year ---
    print(f"  Optimizing Solar+BESS for base year {BASE_YEAR}...")
    try:
        solar_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "Solar")
        bess_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "BESS")

        # Run your optimization function
        cost, solar_cap, bess_energy, lev_cost, _ = optimise_bess(yearly_profile, solar_capex_base, bess_capex_base)

        # Store the result for the base year
        all_results.append({
            "Country": country, "Year": BASE_YEAR, "Tech": "Solar+BESS",
            "LCOE": lev_cost, "Cost": cost, "Solar_Capacity_MW": solar_cap, "BESS_Energy_MWh": bess_energy,
        })
        print(f"  -> Optimal capacity for {country}: Solar={solar_cap:.2f} MW, BESS={bess_energy:.2f} MWh")

    except ValueError as e:
        print(f"  ERROR: Could not optimize for {country} in {BASE_YEAR}. Skipping. Reason: {e}")
        continue  # Skip to the next country if optimization fails

    # --- Step 2: Use fixed capacities to calculate historical Solar+BESS LCOE ---
    print(f"  Calculating historical Solar+BESS LCOE...")
    for year in YEARS:
        if year == BASE_YEAR: continue  # Already calculated this one

        result = calculate_solar_bess_lcoe(country, year, solar_cap, bess_energy, yearly_profile, capex_opex_df)
        if result:
            all_results.append({
                "Country": country, "Year": year, "Tech": "Solar+BESS",
                "LCOE": result.get("LCOE"), "Cost": result.get("Total_Capex"),
                "Solar_Capacity_MW": solar_cap, "BESS_Energy_MWh": bess_energy,
            })

    # --- Step 3: Calculate LCOE for conventional technologies across all years ---
    for tech in CONVENTIONAL_TECHS:
        print(f"  Calculating {tech} LCOE for all years...")
        for year in YEARS:
            result = calculate_conventional_lcoe(country, year, tech, capex_opex_df)
            if result:
                all_results.append({
                    "Country": country, "Year": year, "Tech": tech,
                    "LCOE": result.get("LCOE"), "Cost": None, "Solar_Capacity_MW": None, "BESS_Energy_MWh": None,
                })

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