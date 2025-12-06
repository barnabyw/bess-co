# main_workflow.py
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import logging

# === Import custom modules ===
from Code.data_prep.reader import get_val
from profile import generate_hourly_historical_solar_profile
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
APPEND_RESULTS = True

# === Logging ===
setup_logging(OUTPUT_PATH)
logger = logging.getLogger("main_workflow")

logger.info("Loading input data...")
countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))
logger.info("Data loaded successfully.")

# === Select Countries ===
target_countries = ["United Kingdom"]
countries_to_process = countries_df[countries_df["Country"].isin(target_countries)]

# === Availability sweep ===
AVAILABILITIES = [round(a, 2) for a in np.arange(0.05, 1.001, 0.05)]

# === MAIN WORKFLOW ===
all_results = []

overall_pbar = tqdm(total=len(countries_to_process),
                    desc="Overall Progress",
                    colour="green")

for _, row in countries_to_process.iterrows():

    country = row["Country"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    logger.info("Processing %s (lat=%.4f, lon=%.4f)...", country, lat, lon)

    try:
        yearly_profile = generate_hourly_historical_solar_profile(lat, lon, solar_year=2023)
    except RuntimeError as e:
        logger.error(f"Solar profile generation failed for {country}: {e}")
        continue

    # Base-year CAPEX
    try:
        solar_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "solar")
        bess_energy_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex_e", "BESS")
        bess_power_capex_base  = get_val(capex_opex_df, country, BASE_YEAR, "capex_p", "BESS")
    except ValueError as e:
        logger.error("Missing CAPEX for %s %s: %s", country, BASE_YEAR, e)
        overall_pbar.update(1)
        continue

    # --- AVAILABILITY LOOP ---
    for avail in tqdm(AVAILABILITIES, desc=f"Avail Sweep ({country})", leave=False):

        # === 1) Optimise ===
        try:
            cost, solar_cap, bess_energy, bess_power, results_data, cycles = optimise_bess(
                yearly_profile,
                solar_capex_base,
                bess_energy_capex_base,
                bess_power_capex_base,
                load=1.0,
                availability=avail,
                efficiency=0.9,
                start_soc=0.5
            )

            result_id = len(all_results)

            sb_result = calculate_solar_bess_lcoe(
                country=country,
                year=BASE_YEAR,
                solar_capacity_mw=solar_cap,
                bess_energy_mwh=bess_energy,
                bess_power_mw=bess_power,
                availability=avail,
                bess_cycles=cycles,
                capex_opex_df=capex_opex_df,
                result_id=result_id
            )

            if sb_result is None:
                raise ValueError("Returned None from Solar+BESS LCOE")

            all_results.append({
                "Result_ID": result_id,
                "Country": country,
                "Year": BASE_YEAR,
                "Tech": "Solar+BESS",
                "Availability": avail,
                "LCOE": round(sb_result["LCOE"], 2),
                "Cost": sb_result["Total_Capex"],
                "Solar_Capacity_MW": solar_cap,
                "BESS_Energy_MWh": bess_energy,
                "BESS_Power_MW": bess_power
            })

        except ValueError as e:
            logger.error("Failed Solar+BESS for %s avail %.2f: %s",
                         country, avail, e)
            continue

        # === 2) Historical years ===
        for year in YEARS:
            if year == BASE_YEAR:
                continue
            try:
                rid = len(all_results)

                hist_result = calculate_solar_bess_lcoe(
                    country=country,
                    year=year,
                    solar_capacity_mw=solar_cap,
                    bess_energy_mwh=bess_energy,
                    bess_power_mw=bess_power,
                    availability=avail,
                    bess_cycles=cycles,
                    capex_opex_df=capex_opex_df,
                    result_id=rid
                )

                if hist_result:
                    all_results.append({
                        "Result_ID": rid,
                        "Country": country,
                        "Year": year,
                        "Tech": "Solar+BESS",
                        "Availability": avail,
                        "LCOE": round(hist_result["LCOE"], 2),
                        "Cost": hist_result["Total_Capex"],
                        "Solar_Capacity_MW": solar_cap,
                        "BESS_Energy_MWh": bess_energy,
                        "BESS_Power_MW": bess_power
                    })

            except ValueError:
                continue

        # === 3) conventional techs ===
        for tech in CONVENTIONAL_TECHS:
            for year in YEARS:
                try:
                    rid = len(all_results)
                    conv_result = calculate_conventional_lcoe(
                        country=country,
                        year=year,
                        tech=tech,
                        capacity_mw=1.0,
                        capacity_factor=avail,
                        capex_opex_df=capex_opex_df,
                        result_id=rid
                    )

                    if conv_result:
                        all_results.append({
                            "Result_ID": rid,
                            "Country": country,
                            "Year": year,
                            "Tech": tech,
                            "Availability": avail,
                            "LCOE": conv_result["LCOE"],
                            "Cost": conv_result["Total_Capex"]
                        })

                except ValueError:
                    continue

    overall_pbar.update(1)

overall_pbar.close()

# === SAVE OUTPUT ===
results_df = pd.DataFrame(all_results)
results_df.to_csv(output_file, index=False)
logger.info(f"Saved results to {output_file}")
