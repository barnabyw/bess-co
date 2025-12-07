# main_workflow.py
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import logging

# === Custom modules ===
from Code.data_prep.reader import get_val
from profile import generate_hourly_historical_solar_profile
from optimiser import optimise_bess
from lcoe_helpers import calculate_solar_bess_lcoe, calculate_conventional_lcoe
from logging_conf import setup_logging

# === User selections ===
target_countries = ["Spain"] #"Saudi Arabia", "United Kingdom", "China"]
AVAILABILITIES = [round(a, 2) for a in np.arange(0.05, 1.001, 0.05)]
APPEND_RESULTS = False

# === Paths ===
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

output_file      = os.path.join(OUTPUT_PATH, "lcoe_results.csv")
breakdown_file   = os.path.join(OUTPUT_PATH, "lcoe_breakdowns.csv")
audit_file       = os.path.join(OUTPUT_PATH, "audit_log_unique.csv")

BASE_YEAR = 2024
YEARS = list(range(2015, 2025))
CONVENTIONAL_TECHS = ["Coal", "Gas"]

# === Logging ===
setup_logging(OUTPUT_PATH)
logger = logging.getLogger("main_workflow")

# === AUDIT STORAGE (NEW) ===
used_fallbacks = {}    # <---- Collect all fallback behaviour from get_val()

# === Load data ===
logger.info("Loading inputs…")
countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))
logger.info("Loaded input files successfully.")

# === Automatically select first 50 countries ===
target_countries = countries_df["Country"].iloc[65:120].tolist()
logger.info(f"Processing countries 51–65: {target_countries}")

countries_to_process = countries_df[countries_df["Country"].isin(target_countries)]

# === Storage ===
all_results = []
breakdown_rows = []
errors = []

overall_pbar = tqdm(total=len(countries_to_process), desc="Overall Progress", colour="green")

# =====================================================================
#                            MAIN LOOP
# =====================================================================
for _, row in countries_to_process.iterrows():

    country = row["Country"]
    lat, lon = row["Latitude"], row["Longitude"]

    # --- Solar profile ---
    try:
        yearly_profile = generate_hourly_historical_solar_profile(lat, lon, solar_year=2023)
    except Exception as e:
        msg = f"Solar profile generation FAILED for {country}: {e}"
        logger.error(msg)
        errors.append(msg)
        continue

    # --- Base-year CAPEX ---
    try:
        solar_capex_base = get_val(
            capex_opex_df, country, BASE_YEAR, "capex", "solar",
            used_fallbacks=used_fallbacks
        )
        bess_energy_capex_base = get_val(
            capex_opex_df, country, BASE_YEAR, "capex_e", "BESS",
            used_fallbacks=used_fallbacks
        )
        bess_power_capex_base = get_val(
            capex_opex_df, country, BASE_YEAR, "capex_p", "BESS",
            used_fallbacks=used_fallbacks
        )
    except Exception as e:
        msg = f"CAPEX lookup FAILED for {country} {BASE_YEAR}: {e}"
        logger.error(msg)
        errors.append(msg)
        overall_pbar.update(1)
        continue

    # === AVAIL LOOP ===
    for avail in tqdm(AVAILABILITIES, desc=f"{country}", leave=False):

        # ---------------------------------------------------------
        # Solar + BESS optimisation for BASE YEAR
        # ---------------------------------------------------------
        try:
            cost, solar_cap, bess_energy, bess_power, results_data, cycles = optimise_bess(
                yearly_profile,
                solar_capex_base,
                bess_energy_capex_base,
                bess_power_capex_base,
                load=1.0,
                availability=avail,
                efficiency=0.9,
                start_soc=0.5,
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
                result_id=result_id,
                used_fallbacks=used_fallbacks
            )

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
                "BESS_Power_MW": bess_power,
            })

            for comp, row_b in sb_result["Breakdown"].iterrows():
                breakdown_rows.append({
                    "Country": country,
                    "Year": BASE_YEAR,
                    "Tech": "Solar+BESS",
                    "Availability": avail,
                    "Component": comp,
                    "Value": row_b["Value"],
                })

        except Exception as e:
            msg = f"S+BESS FAILED for {country} avail={avail}: {e}"
            logger.error(msg)
            errors.append(msg)
            continue

        # ---------------------------------------------------------
        #   Historical Years (Solar + BESS)
        # ---------------------------------------------------------
        for year in YEARS:
            if year == BASE_YEAR:
                continue

            try:
                rid = len(all_results)

                hist = calculate_solar_bess_lcoe(
                    country=country,
                    year=year,
                    solar_capacity_mw=solar_cap,
                    bess_energy_mwh=bess_energy,
                    bess_power_mw=bess_power,
                    availability=avail,
                    bess_cycles=cycles,
                    capex_opex_df=capex_opex_df,
                    result_id=rid,
                    used_fallbacks=used_fallbacks
                )

                all_results.append({
                    "Result_ID": rid,
                    "Country": country,
                    "Year": year,
                    "Tech": "Solar+BESS",
                    "Availability": avail,
                    "LCOE": round(hist["LCOE"], 2),
                    "Cost": hist["Total_Capex"],
                    "Solar_Capacity_MW": solar_cap,
                    "BESS_Energy_MWh": bess_energy,
                    "BESS_Power_MW": bess_power,
                })

                for comp, row_b in hist["Breakdown"].iterrows():
                    breakdown_rows.append({
                        "Country": country,
                        "Year": year,
                        "Tech": "Solar+BESS",
                        "Availability": avail,
                        "Component": comp,
                        "Value": row_b["Value"],
                    })

            except Exception as e:
                msg = f"S+BESS historical FAILED for {country}, {year}, avail={avail}: {e}"
                logger.error(msg)
                errors.append(msg)
                continue

        # ---------------------------------------------------------
        #   Conventional Techs
        # ---------------------------------------------------------
        for tech in CONVENTIONAL_TECHS:
            for year in YEARS:
                try:
                    rid = len(all_results)
                    conv = calculate_conventional_lcoe(
                        country=country,
                        year=year,
                        tech=tech,
                        capacity_mw=1.0,
                        capacity_factor=avail,
                        capex_opex_df=capex_opex_df,
                        result_id=rid,
                        used_fallbacks=used_fallbacks,
                    )
                    all_results.append({
                        "Result_ID": rid,
                        "Country": country,
                        "Year": year,
                        "Tech": tech,
                        "Availability": avail,
                        "LCOE": conv["LCOE"],
                        "Cost": conv["Total_Capex"],
                    })

                    for comp, row_b in conv["Breakdown"].iterrows():
                        breakdown_rows.append({
                            "Country": country,
                            "Year": year,
                            "Tech": tech,
                            "Availability": avail,
                            "Component": comp,
                            "Value": row_b["Value"],
                        })

                except Exception as e:
                    msg = f"{tech} FAILED for {country}, {year}, avail={avail}: {e}"
                    logger.error(msg)
                    errors.append(msg)
                    continue

    overall_pbar.update(1)

overall_pbar.close()

# =====================================================================
#                       SAVE OUTPUTS
# =====================================================================
results_df = pd.DataFrame(all_results)
breakdown_df = pd.DataFrame(breakdown_rows)

if APPEND_RESULTS and os.path.exists(output_file):
    results_df = pd.concat([pd.read_csv(output_file), results_df], ignore_index=True)
if APPEND_RESULTS and os.path.exists(breakdown_file):
    breakdown_df = pd.concat([pd.read_csv(breakdown_file), breakdown_df], ignore_index=True)

results_df.to_csv(output_file, index=False)
breakdown_df.to_csv(breakdown_file, index=False)

# === SAVE FALLBACK AUDIT FILE ===
audit_df = pd.DataFrame([
    {
        "Country": k[0],
        "Variable": k[1],
        "Tech": k[2],
        "Year": k[3],
        "Region_Used": v,
    }
    for k, v in used_fallbacks.items()
])
# === SAVE FALLBACK AUDIT FILE ===
if APPEND_RESULTS and os.path.exists(audit_file):
    old_audit = pd.read_csv(audit_file)
    audit_df = pd.concat([old_audit, audit_df], ignore_index=True)

audit_df.to_csv(audit_file, index=False)

# =====================================================================
#                     COMPLETION SUMMARY
# =====================================================================
print("\n===================== RUN SUMMARY =====================")
print(f"Total results rows:          {len(results_df):,}")
print(f"Total breakdown rows:        {len(breakdown_df):,}")
print(f"Fallback audit entries:      {len(audit_df):,}")
print(f"Total errors encountered:    {len(errors):,}")

if len(errors) > 0:
    print("\n--- First few errors ---")
    for e in errors[:5]:
        print(" •", e)

print("\nOutputs saved to:")
print("  →", output_file)
print("  →", breakdown_file)
print("  →", audit_file)
print("========================================================\n")
