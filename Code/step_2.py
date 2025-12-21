# main_calculate_lcoe.py
import pandas as pd
import os
from tqdm import tqdm
import logging

from Code.data_prep.reader import get_val
from Code.lcoe_helpers import calculate_solar_bess_lcoe, calculate_conventional_lcoe
from Code.logging_conf import setup_logging

# =========================================================
# Configuration
# =========================================================
BASE_YEAR = 2024
YEARS = list(range(2015, 2031))
FOSSIL_CAPACITY_FACTORS = [0.6, 0.7, 0.8]
FOSSIL_PLOT_AVAILS = [0.05, 0.7, 1.0]
CONVENTIONAL_TECHS = ["Coal", "Gas"]

APPEND_RESULTS = True

# Paths
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")
OUTPUT_PATH = os.path.join(CWD, "..", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

sizes_file = os.path.join(OUTPUT_PATH, "solar_bess_sizes.csv")
capex_opex_file = os.path.join(INPUT_PATH, "capex_opex_converted.xlsx")

output_file = os.path.join(OUTPUT_PATH, "lcoe_results.csv")
breakdown_file = os.path.join(OUTPUT_PATH, "lcoe_breakdowns2.csv")
audit_file = os.path.join(OUTPUT_PATH, "audit_log_unique.csv")

# Logging
setup_logging(OUTPUT_PATH)
logger = logging.getLogger("main_calculate_lcoe")

used_fallbacks = {}

# =========================================================
# Load inputs
# =========================================================
sizes_df = pd.read_csv(sizes_file)
capex_opex_df = pd.read_excel(capex_opex_file)

# =========================================================
# Main loop
# =========================================================

first_avail_by_country = (
    sizes_df
    .groupby("Country")["Availability"]
    .min()
    .to_dict()
)


all_results = []
breakdown_rows = []
grouped = sizes_df.groupby(["Country", "Availability"])

for (country, avail), grp in tqdm(grouped, desc="Countries × Availability"):

    row = grp.iloc[0]

    solar_cap = row["Solar_Capacity_MW"]
    bess_energy = row["BESS_Energy_MWh"]
    bess_power = row["BESS_Power_MW"]



    # -----------------------------------------------------
    # Solar + BESS LCOE (all years)  ← unchanged
    # -----------------------------------------------------
    for year in YEARS:
        try:
            rid = len(all_results)

            sb = calculate_solar_bess_lcoe(
                country=country,
                year=year,
                solar_capacity_mw=solar_cap,
                bess_energy_mwh=bess_energy,
                bess_power_mw=bess_power,
                availability=avail,
                capex_opex_df=capex_opex_df,
                discount_rate=0.08,
                lifetime=25,
                result_id=rid,
                used_fallbacks=used_fallbacks,
                bess_cycles=300,
            )

            all_results.append({
                "Result_ID": rid,
                "Country": country,
                "Year": year,
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
                    "Year": year,
                    "Tech": "Solar+BESS",
                    "Availability": avail,
                    "Component": comp,
                    "Value": row_b["Value"],
                })

        except Exception as e:
            logger.error(f"S+BESS failed for {country}, {year}, {avail}: {e}")

    # -----------------------------------------------------
    # Conventional techs (RUN ONCE PER COUNTRY)
    # -----------------------------------------------------
    if avail != first_avail_by_country[country]:
        continue

    for tech in CONVENTIONAL_TECHS:
        for cf in FOSSIL_CAPACITY_FACTORS:  # 👈 NEW
            for year in YEARS:
                try:
                    rid = len(all_results)

                    conv = calculate_conventional_lcoe(
                        country=country,
                        year=year,
                        tech=tech,
                        capacity_mw=1.0,
                        capacity_factor=cf,
                        capex_opex_df=capex_opex_df,
                        discount_rate=0.08,
                        lifetime=25,
                        result_id=rid,
                        used_fallbacks=used_fallbacks,
                    )

                    for plot_avail in FOSSIL_PLOT_AVAILS:
                        all_results.append({
                            "Result_ID": rid,
                            "Country": country,
                            "Year": year,
                            "Tech": tech,
                            "Availability": plot_avail,
                            "LCOE": conv["LCOE"],
                            "Cost": conv["Total_Capex"],
                            "Capacity Factor": cf,  # 👈 stored explicitly
                        })

                        for comp, row_b in conv["Breakdown"].iterrows():
                            breakdown_rows.append({
                                "Country": country,
                                "Year": year,
                                "Tech": tech,
                                "Availability": plot_avail,
                                "Component": comp,
                                "Value": row_b["Value"],
                                "Capacity Factor": cf,
                            })

                except Exception as e:
                    logger.error(f"{tech} FAILED for {country}, {year}, CF={cf}: {e}")

# =========================================================
# Save outputs
# =========================================================
pd.DataFrame(all_results).to_csv(output_file, index=False)
pd.DataFrame(breakdown_rows).to_csv(breakdown_file, index=False)

audit_rows = [
    {"Country": k[0], "Variable": k[1], "Tech": k[2], "Year": k[3], "Region_Used": v}
    for k, v in used_fallbacks.items()
]
pd.DataFrame(audit_rows).to_csv(audit_file, index=False)

print("Done.")
