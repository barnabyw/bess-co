import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------------
# Config – update these paths/names to match your setup
# -------------------------------------------------------------------
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "inputs")

ext_inputs = r"C:\Users\barna\OneDrive\Documents\Solar_BESS\inputs"
ext_import = r"C:\Users\barna\OneDrive\Documents\Solar_BESS\inputs\raw"

WORLD_FILE = os.path.join(ext_import, "world_bess_capex_raw.csv")
MODO_FILE  = os.path.join(ext_import, "modo_bess_costs.csv")
OUTPUT_FILE = os.path.join(ext_import, "world_bess_capex_split_pe.csv")

# Assumptions
DURATION_HOURS = 4.0
R_POWER_ENERGY = 4.0


# -------------------------------------------------------------------
# Split historic World capex into energy/power components
# -------------------------------------------------------------------
def split_world_capex_energy_power(world_df):
    mask = (world_df["variable"] == "capex") & (world_df["tech"] == "BESS")
    capex_df = world_df[mask].copy()

    reported = capex_df["value"].astype(float)

    d = DURATION_HOURS
    r = R_POWER_ENERGY

    capex_e = reported * d / (r + d)
    capex_p = r * capex_e

    capex_df["capex_e"] = capex_e
    capex_df["capex_p"] = capex_p

    return capex_df


# -------------------------------------------------------------------
# Compute YoY decline for ALL Modo scenarios
# -------------------------------------------------------------------
def compute_modo_decline_factors(modo_df):

    modo_df = modo_df.rename(columns={"Start Year": "year"})
    modo_df["year"] = modo_df["year"].astype(int)

    out = {}

    for scenario in modo_df["Version"].unique():
        df_s = modo_df[modo_df["Version"] == scenario].copy()
        df_s = df_s.sort_values("year").reset_index(drop=True)

        years = df_s["year"].tolist()
        per_e = df_s["per kwh"].tolist()
        per_p = df_s["pw kw"].tolist()

        rows = []
        for i in range(1, len(years)):
            rows.append({
                "year": years[i],
                "yoy_e": per_e[i] / per_e[i-1],
                "yoy_p": per_p[i] / per_p[i-1],
            })

        out[scenario] = pd.DataFrame(rows)

    return out   # dict: {scenario → DataFrame}


# -------------------------------------------------------------------
# Project World CAPEX for a given scenario
# -------------------------------------------------------------------
def project_world_from_modo(world_split, modo_yoy_df, scenario_name):
    base_yr = 2024
    base_row = world_split.loc[world_split["year"] == base_yr].iloc[0]

    base_e = base_row["capex_e"]
    base_p = base_row["capex_p"]

    proj_rows = []

    # STEP 1 — 2025
    yoy_2026 = modo_yoy_df.loc[modo_yoy_df["year"] == 2026].iloc[0]
    world_e_2025 = base_e * yoy_2026["yoy_e"]
    world_p_2025 = base_p * yoy_2026["yoy_p"]

    proj_rows.append({
        "year": 2025,
        "capex_e": world_e_2025,
        "capex_p": world_p_2025,
        "scenario": scenario_name
    })

    # STEP 2 — Iterate forward
    prev_e, prev_p = world_e_2025, world_p_2025

    for _, row in modo_yoy_df.iterrows():
        year = int(row["year"])
        if year <= 2025:
            continue

        next_e = prev_e * row["yoy_e"]
        next_p = prev_p * row["yoy_p"]

        proj_rows.append({
            "year": year,
            "capex_e": round(next_e, 1),
            "capex_p": round(next_p, 1),
            "scenario": scenario_name
        })

        prev_e, prev_p = next_e, next_p

    proj_df = pd.DataFrame(proj_rows)
    proj_df["region"] = "World"
    proj_df["tech"] = "BESS"
    proj_df["variable"] = "capex_p"
    proj_df["units"] = "kWh"
    proj_df["type"] = "projected"
    proj_df["money"] = "USD"
    proj_df["money year"] = 2024

    return proj_df


# -------------------------------------------------------------------
# Build long-format CAPEX + OPEX
# -------------------------------------------------------------------
def build_long_capex(world_hist, world_proj_all, world_raw):

    all_capex = pd.concat([world_hist, world_proj_all], ignore_index=True)

    rows = []

    for _, row in all_capex.iterrows():

        # energy CAPEX
        rows.append({
            "year": row["year"],
            "scenario": row.get("scenario", ""),
            "region": "World",
            "tech": "BESS",
            "variable": "capex_e",
            "value": row["capex_e"],
            "units": "kWh",
            "money": "USD",
            "money year": 2024,
            "type": row.get("type", "historic"),
            "source": row.get("source", "Derived")
        })

        # power CAPEX
        rows.append({
            "year": row["year"],
            "scenario": row.get("scenario", ""),
            "region": "World",
            "tech": "BESS",
            "variable": "capex_p",
            "value": row["capex_p"],
            "units": "kW",
            "money": "USD",
            "money year": 2024,
            "type": row.get("type", "historic"),
            "source": row.get("source", "Derived")
        })

    # OPEX (single row, scenario empty)
    opex_row = world_raw[(world_raw["variable"] == "opex_f") &
                         (world_raw["tech"] == "BESS")].iloc[0]

    rows.append({
        "year": "all",
        "scenario": "",
        "region": opex_row["region"],
        "tech": "BESS",
        "variable": "opex_f",
        "value": opex_row["value"],
        "units": opex_row["units"],
        "money": opex_row["money"],
        "money year": opex_row["money year"],
        "type": opex_row["type"],
        "source": opex_row["source"]
    })

    df_out = pd.DataFrame(rows)

    # SORT
    df_out["year_sort"] = df_out["year"].replace("all", 9999).astype(int)
    df_out = df_out.sort_values(["scenario", "variable", "year_sort"])
    df_out = df_out.drop(columns=["year_sort"])

    return df_out


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # Load raw inputs for world projections
    world_raw = pd.read_csv(WORLD_FILE)
    modo_raw  = pd.read_csv(MODO_FILE)

    world_raw["year"] = world_raw["year"].replace("all", np.nan)
    world_raw["year"] = world_raw["year"].astype(float)

    # === Build historical ===
    world_hist = split_world_capex_energy_power(world_raw)
    world_hist["scenario"] = ""   # historical always empty

    # === Compute YoY decline for all Modo scenarios ===
    modo_yoy_dict = compute_modo_decline_factors(modo_raw)

    # === Project central scenario (scenario empty string) ===
    proj_central = project_world_from_modo(
        world_hist, modo_yoy_dict["Central (v3.5)"], scenario_name=""
    )

    # === Project low scenario ===
    proj_low = project_world_from_modo(
        world_hist, modo_yoy_dict["Faster reduction"], scenario_name="Low"
    )

    # === Combine projections ===
    proj_all = pd.concat([proj_central, proj_low], ignore_index=True)

    # === Build final long-format BESS capex/opex ===
    df_final = build_long_capex(world_hist, proj_all, world_raw)

    # -------------------------------------------------------------------
    # NEW: Append df_final to capex_opex.xlsx and save capex_opex_2.xlsx
    # -------------------------------------------------------------------
    capex_opex_path = os.path.join(ext_inputs, "capex_opex.xlsx")
    capex_opex_df = pd.read_excel(capex_opex_path, sheet_name="capex_opex")

    # Make columns consistent (ensure missing columns exist)
    for col in capex_opex_df.columns:
        if col not in df_final.columns:
            df_final[col] = np.nan
    df_final = df_final[list(capex_opex_df.columns)]

    # Append
    combined = pd.concat([capex_opex_df, df_final], ignore_index=True)

    # Save as capex_opex_2.xlsx
    output_path = os.path.join(ext_inputs, "capex_opex_2.xlsx")
    combined.to_excel(output_path, index=False)

    print("\n----- COMPLETE -----")
    print("World projections file saved →", OUTPUT_FILE)
    print("capex_opex with appended World rows saved →", output_path)
    print("\nPreview of added rows:")
    print(df_final.head(10))

if __name__ == "__main__":
    main()
