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
#target_countries = ["Australia", "United Kingdom", "China", "Brazil", "Russia", "India"]  # Full list overridden below after loading file
AVAILABILITIES = [round(a, 2) for a in np.arange(0.05, 1.001, 0.05)]
APPEND_RESULTS = True  # Append across runs? True = overwrite matching rows

# === Paths ===
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

output_file = os.path.join(OUTPUT_PATH, "lcoe_results.csv")
breakdown_file = os.path.join(OUTPUT_PATH, "lcoe_breakdowns.csv")
audit_file = os.path.join(OUTPUT_PATH, "audit_log_unique.csv")

BASE_YEAR = 2024
YEARS = list(range(2015, 2025))
CONVENTIONAL_TECHS = ["Coal", "Gas"]

# === Logging ===
setup_logging(OUTPUT_PATH)
logger = logging.getLogger("main_workflow")

# === AUDIT STORAGE ===
used_fallbacks = {}

# === Load Inputs ===
logger.info("Loading inputs…")
countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))
logger.info("Loaded input files successfully.")

# === Determine remaining countries (restart-safe) ===
target_countries = countries_df["Country"].tolist()

completed_countries = set()
if os.path.exists(output_file):
    try:
        existing_df = pd.read_csv(output_file, usecols=["Country"])
        completed_countries = set(existing_df["Country"].unique())
        print(f"Restart-safe: {len(completed_countries)} countries already completed.")
    except Exception as e:
        print("WARNING: Could not read existing output for restart:", e)

remaining_countries = [c for c in target_countries if c not in completed_countries]

countries_to_process = countries_df[countries_df["Country"].isin(remaining_countries)]
print(f"Restart-safe: {len(remaining_countries)} remaining countries to process.")

# === Keys for overwrite logic ===
KEY_COLS_RESULTS = ["Country", "Year", "Availability", "Tech"]
KEY_COLS_BREAKDOWN = ["Country", "Year", "Availability", "Tech", "Component"]
KEY_COLS_AUDIT = ["Country", "Variable", "Tech", "Year"]

# ============================================================
# Write helpers
# ============================================================
def append_with_overwrite(new_df: pd.DataFrame, path: str, key_cols: list[str]):
    """Append new_df to CSV, overwriting rows with matching primary keys."""
    if new_df.empty:
        return
    if not os.path.exists(path):
        new_df.to_csv(path, index=False)
        return

    old = pd.read_csv(path)

    keys_df = new_df[key_cols].drop_duplicates()
    merged = old.merge(keys_df, on=key_cols, how="left", indicator=True)
    filtered_old = merged[merged["_merge"] == "left_only"]
    filtered_old = filtered_old[old.columns]

    final = pd.concat([filtered_old, new_df], ignore_index=True)
    final.to_csv(path, index=False)


def append_no_overwrite(new_df: pd.DataFrame, path: str):
    """Append to CSV without any overwriting (fresh run only)."""
    if new_df.empty:
        return
    write_header = not os.path.exists(path)
    new_df.to_csv(path, mode="a", header=write_header, index=False)


# Clear outputs at the beginning **only** when APPEND_RESULTS=False
if not APPEND_RESULTS:
    for f in [output_file, breakdown_file, audit_file]:
        if os.path.exists(f):
            os.remove(f)

# === Local storage while computing a country ===
all_results = []
breakdown_rows = []
errors = []

overall_pbar = tqdm(total=len(countries_to_process), desc="Overall Progress", colour="green")

# =====================================================================
#                               MAIN LOOP
# =====================================================================
for _, row in countries_to_process.iterrows():

    country = row["Country"]
    lat, lon = row["Latitude"], row["Longitude"]

    # --- Solar profile ---
    try:
        yearly_profile = generate_hourly_historical_solar_profile(lat, lon, solar_year=2023)
    except Exception as e:
        msg = f"Solar profile FAILED for {country}: {e}"
        logger.error(msg)
        errors.append(msg)
        overall_pbar.update(1)
        continue

    # --- CAPEX lookups ---
    try:
        solar_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex", "solar", used_fallbacks=used_fallbacks)
        bess_energy_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex_e", "BESS", used_fallbacks=used_fallbacks)
        bess_power_capex_base = get_val(capex_opex_df, country, BASE_YEAR, "capex_p", "BESS", used_fallbacks=used_fallbacks)
    except Exception as e:
        msg = f"CAPEX lookup FAILED for {country}: {e}"
        logger.error(msg)
        errors.append(msg)
        overall_pbar.update(1)
        continue

    # === Availability loop ===
    for avail in tqdm(AVAILABILITIES, desc=f"{country}", leave=False):

        # ---------------------------------------------------------
        # Base year optimisation
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

            rid = len(all_results)
            sb = calculate_solar_bess_lcoe(
                country=country,
                year=BASE_YEAR,
                solar_capacity_mw=solar_cap,
                bess_energy_mwh=bess_energy,
                bess_power_mw=bess_power,
                availability=avail,
                bess_cycles=cycles,
                capex_opex_df=capex_opex_df,
                discount_rate=0.08,
                lifetime=25,
                result_id=rid,
                used_fallbacks=used_fallbacks,
            )

            all_results.append({
                "Result_ID": rid,
                "Country": country,
                "Year": BASE_YEAR,
                "Tech": "Solar+BESS",
                "Availability": avail,
                "LCOE": round(sb["LCOE"], 2),
                "Cost": sb["Total_Capex"],
                "Solar_Capacity_MW": solar_cap,
                "BESS_Energy_MWh": bess_energy,
                "BESS_Power_MW": bess_power,
            })

            for comp, row_b in sb["Breakdown"].iterrows():
                breakdown_rows.append({
                    "Country": country,
                    "Year": BASE_YEAR,
                    "Tech": "Solar+BESS",
                    "Availability": avail,
                    "Component": comp,
                    "Value": row_b["Value"],
                })

        except Exception as e:
            msg = f"S+BESS FAILED for {country}, avail={avail}: {e}"
            logger.error(msg)
            errors.append(msg)
            continue

        # ---------------------------------------------------------
        # Historical years
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
                    discount_rate=0.08,
                    lifetime=25,
                    result_id=rid,
                    used_fallbacks=used_fallbacks,
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
                msg = f"S+BESS historical FAILED for {country}, {year}: {e}"
                logger.error(msg)
                errors.append(msg)
                continue

        # ---------------------------------------------------------
        # Conventional techs
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
                        discount_rate=0.08,
                        lifetime=25,
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
                    msg = f"{tech} FAILED for {country}, {year}: {e}"
                    logger.error(msg)
                    errors.append(msg)
                    continue

    # ==========================================================
    # Save completed country
    # ==========================================================
    country_results_df = pd.DataFrame(all_results)
    country_breakdown_df = pd.DataFrame(breakdown_rows)

    if APPEND_RESULTS:
        append_with_overwrite(country_results_df, output_file, KEY_COLS_RESULTS)
        append_with_overwrite(country_breakdown_df, breakdown_file, KEY_COLS_BREAKDOWN)
    else:
        append_no_overwrite(country_results_df, output_file)
        append_no_overwrite(country_breakdown_df, breakdown_file)

    all_results.clear()
    breakdown_rows.clear()

    # Save audit log
    audit_rows = [
        {"Country": k[0], "Variable": k[1], "Tech": k[2], "Year": k[3], "Region_Used": v}
        for k, v in used_fallbacks.items()
        if k[0] == country
    ]
    audit_df_country = pd.DataFrame(audit_rows)

    if APPEND_RESULTS:
        append_with_overwrite(audit_df_country, audit_file, KEY_COLS_AUDIT)
    else:
        append_no_overwrite(audit_df_country, audit_file)

    # Remove written entries
    used_fallbacks = {k: v for k, v in used_fallbacks.items() if k[0] != country}

    overall_pbar.update(1)

overall_pbar.close()

# =============================================================================
# Final summary only (no saving here!)
# =============================================================================
print("\n===================== RUN SUMMARY =====================")
if os.path.exists(output_file):
    print("Total results rows:          ", len(pd.read_csv(output_file)))
if os.path.exists(breakdown_file):
    print("Total breakdown rows:        ", len(pd.read_csv(breakdown_file)))
if os.path.exists(audit_file):
    print("Fallback audit entries:      ", len(pd.read_csv(audit_file)))

print("\nErrors:", len(errors))
if errors:
    print("First few:")
    for e in errors[:5]:
        print(" •", e)

print("\nOutputs saved to:")
print("  →", output_file)
print("  →", breakdown_file)
print("  →", audit_file)
print("========================================================\n")
