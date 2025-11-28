# main_workflow.py
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import logging

# === Import custom modules ===
from reader import get_val
from profile import generate_hourly_solar_profile
from optimiser import optimise_bess
from lcoe_helpers import calculate_solar_bess_lcoe, calculate_conventional_lcoe
from logging_conf import setup_logging

# === Configuration ===
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)
output_file = os.path.join(OUTPUT_PATH, "lcoe_results.csv")

BASE_YEAR = 2024
YEARS = list(range(2015, 2025))
CONVENTIONAL_TECHS = ["Coal", "Gas"]

# === Logging ===
setup_logging(OUTPUT_PATH)
logger = logging.getLogger("main_workflow")

# === Global audit store ===
AUDIT_LOG = []

# === Load Data ===
logger.info("Loading input data...")
countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))
logger.info("Data loaded successfully.")

# === Select Countries ===
target_countries = ["United States", "Australia", "Spain"] #,"Saudi Arabia", "Chile", "United Kingdom"]

if target_countries:
    countries_to_process = countries_df[countries_df["Country"].isin(target_countries)]
    logger.info(
        "Running analysis for %d selected countries: %s",
        len(countries_to_process),
        ", ".join(target_countries)
    )
else:
    countries_to_process = countries_df
    logger.info("Running analysis for all %d countries.", len(countries_to_process))

# === Availability sweep ===
AVAILABILITIES = [round(a, 2) for a in np.arange(0.05, 1.001, 0.05)]

# === MAIN WORKFLOW ===
all_results = []

# -----------------------------------------------------------
#                OVERALL PROGRESS BAR (NEW)
# -----------------------------------------------------------
overall_pbar = tqdm(
    total=len(countries_to_process),
    desc="Overall Progress",
    colour="green"
)

for _, row in countries_to_process.iterrows():

    country = row["Country"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    logger.info("Processing %s (lat=%.4f, lon=%.4f)...", country, lat, lon)

    # Generate solar profile once per country
    yearly_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)

    # Base-year CAPEX (audited)
    try:
        solar_capex_base = get_val(
            capex_opex_df, country, BASE_YEAR, "capex", "solar",
            audit_log=AUDIT_LOG,
            audit_context={"phase": "base_capex", "country": country}
        )
        bess_capex_base = get_val(
            capex_opex_df, country, BASE_YEAR, "capex", "bess",
            audit_log=AUDIT_LOG,
            audit_context={"phase": "base_capex", "country": country}
        )
    except ValueError as e:
        logger.error("Missing CAPEX for %s %s: %s", country, BASE_YEAR, e)
        overall_pbar.update(1)
        continue

    # -----------------------------------------------------------
    #         AVAILABILITY SWEEP BAR (per country)
    # -----------------------------------------------------------
    for avail in tqdm(AVAILABILITIES, desc=f"Avail Sweep ({country})", leave=False):

        # Step 1 — optimise Solar+BESS
        try:
            cost, solar_cap, bess_energy, _ = optimise_bess(
                yearly_profile, solar_capex_base, bess_capex_base, availability=avail
            )

            result_id = len(all_results)

            sb_result = calculate_solar_bess_lcoe(
                country, BASE_YEAR, solar_cap, bess_energy, avail, capex_opex_df,
                audit_log=AUDIT_LOG,
                result_id=result_id,
            )

            if sb_result is None:
                raise ValueError("Returned None from Solar+BESS LCOE")

            all_results.append({
                "Result_ID": result_id,
                "Country": country, "Year": BASE_YEAR, "Tech": "Solar+BESS",
                "Availability": avail,
                "LCOE": round(sb_result["LCOE"], 2),
                "Cost": sb_result["Total_Capex"],
                "Solar_Capacity_MW": round(solar_cap, 1),
                "BESS_Energy_MWh": round(bess_energy, 1),
            })

            logger.info(
                "%-15s | Avail=%.2f | LCOE=%.2f | Solar=%.1f MW | BESS=%.1f MWh",
                country, avail, sb_result["LCOE"], solar_cap, bess_energy
            )

        except ValueError as e:
            logger.error("Failed Solar+BESS for %s avail %.2f: %s", country, avail, e)
            continue

        # Step 2 — historical Solar+BESS
        for year in YEARS:
            if year == BASE_YEAR:
                continue
            try:
                rid = len(all_results)
                hist_result = calculate_solar_bess_lcoe(
                    country, year, solar_cap, bess_energy, avail, capex_opex_df,
                    audit_log=AUDIT_LOG,
                    result_id=rid,
                )

                if hist_result:
                    all_results.append({
                        "Result_ID": rid,
                        "Country": country, "Year": year, "Tech": "Solar+BESS",
                        "Availability": avail,
                        "LCOE": round(hist_result["LCOE"], 2),
                        "Cost": hist_result["Total_Capex"],
                        "Solar_Capacity_MW": round(solar_cap, 1),
                        "BESS_Energy_MWh": round(bess_energy, 1),
                    })

            except ValueError as e:
                logger.warning(
                    "Skipping Solar+BESS hist %s @ Avail %.2f for %s: %s",
                    year, avail, country, e
                )

        # Step 3 — conventional techs
        for tech in CONVENTIONAL_TECHS:
            for year in YEARS:
                try:
                    cf = avail
                    rid = len(all_results)

                    conv_result = calculate_conventional_lcoe(
                        country=country,
                        year=year,
                        tech=tech,
                        capacity_mw=1.0,
                        capacity_factor=cf,
                        capex_opex_df=capex_opex_df,
                        audit_log=AUDIT_LOG,
                        result_id=rid,
                    )

                    if conv_result:
                        all_results.append({
                            "Result_ID": rid,
                            "Country": country, "Year": year, "Tech": tech,
                            "Availability": avail,
                            "LCOE": conv_result["LCOE"],
                            "Cost": conv_result["Total_Capex"],
                            "Solar_Capacity_MW": None,
                            "BESS_Energy_MWh": None,
                        })

                except ValueError as e:
                    logger.warning(
                        "Skipping %s %s @ Avail %.2f for %s: %s",
                        tech, year, avail, country, e
                    )

    # → Mark country as completed
    overall_pbar.update(1)

overall_pbar.close()

# === Save Outputs ===
logger.info("Analysis complete. Compiling and saving results...")

results_df = pd.DataFrame(all_results)

output_cols = [
    "Result_ID",
    "Country", "Year", "Tech", "Availability",
    "LCOE", "Cost",
    "Solar_Capacity_MW", "BESS_Energy_MWh",
]

results_df = results_df[output_cols]
results_df.to_csv(output_file, index=False)

audit_df = pd.DataFrame(AUDIT_LOG)
audit_file = os.path.join(OUTPUT_PATH, "lcoe_input_audit_log.csv")
#audit_df.to_csv(audit_file, index=False)

logger.info("Results saved to %s", output_file)
logger.info("Audit log saved to %s", audit_file)
logger.info("Head of results:\n%s", results_df.head())
