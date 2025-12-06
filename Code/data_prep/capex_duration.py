import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------------
# Config – update these paths/names to match your setup
# -------------------------------------------------------------------
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "inputs")

ext_import = r"C:\Users\barna\OneDrive\Documents\Solar_BESS\inputs\raw"

WORLD_FILE = os.path.join(ext_import, "world_bess_capex_raw.csv")
MODO_FILE  = os.path.join(ext_import, "modo_bess_costs.csv")
OUTPUT_FILE = os.path.join(ext_import, "world_bess_capex_split_pe.csv")

# Assumptions
DURATION_HOURS = 4.0       # 4h system
R_POWER_ENERGY = 4.0       # Cp / Ce


# -------------------------------------------------------------------
# Split historic World capex into energy/power components
# -------------------------------------------------------------------
def split_world_capex_energy_power(world_df: pd.DataFrame) -> pd.DataFrame:

    mask_capex = (world_df["variable"] == "capex") & (world_df["tech"] == "BESS")
    capex_df = world_df[mask_capex].copy()

    reported = capex_df["value"].astype(float)

    d = DURATION_HOURS
    r = R_POWER_ENERGY

    capex_e = reported * d / (r + d)
    capex_p = r * capex_e

    capex_df["capex_e"] = capex_e
    capex_df["capex_p"] = capex_p

    return capex_df


# -------------------------------------------------------------------
# Compute year-over-year decline factors (2026/2025, 2027/2026, …)
# -------------------------------------------------------------------
def compute_modo_decline_factors(modo_df: pd.DataFrame) -> pd.DataFrame:

    df = modo_df.copy()
    df = df[df["Version"] == "Central (v3.5)"]

    df = df.rename(columns={"Start Year": "year"})
    df["year"] = df["year"].astype(int)

    df = df[["year", "pw kw", "per kwh"]].copy()
    df = df.sort_values("year").reset_index(drop=True)

    years = df["year"].tolist()
    per_e = df["per kwh"].tolist()
    per_p = df["pw kw"].tolist()

    out_rows = []

    for i in range(1, len(years)):
        yoy_e = per_e[i] / per_e[i-1]
        yoy_p = per_p[i] / per_p[i-1]

        out_rows.append({
            "year": years[i],
            "yoy_e": yoy_e,
            "yoy_p": yoy_p
        })

    return pd.DataFrame(out_rows)


# -------------------------------------------------------------------
# Project World future CAPEX using Modo YoY decline, Option A
# -------------------------------------------------------------------
def project_world_from_modo(world_split: pd.DataFrame,
                            modo_yoy: pd.DataFrame) -> pd.DataFrame:

    base_yr = 2024
    base_row = world_split.loc[world_split["year"] == base_yr].iloc[0]
    base_e = base_row["capex_e"]
    base_p = base_row["capex_p"]

    proj_rows = []

    # STEP 1 — compute 2025 using Modo 2026/2025 YoY decline
    yoy_2026 = modo_yoy.loc[modo_yoy["year"] == 2026].iloc[0]

    world_e_2025 = base_e * yoy_2026["yoy_e"]
    world_p_2025 = base_p * yoy_2026["yoy_p"]

    proj_rows.append({
        "year": 2025,
        "capex_e": world_e_2025,
        "capex_p": world_p_2025,
        "region": "World",
        "tech": "BESS",
        "variable": "capex",
        "units": "kWh",
        "type": "projected",
        "money": "USD",
        "money year": 2024,
        "source": "Derived from 2026/2025 Modo YoY"
    })

    # STEP 2 — iterate from 2026 onward
    prev_e = world_e_2025
    prev_p = world_p_2025

    for _, row in modo_yoy.iterrows():
        year = int(row["year"])
        if year <= 2025:
            continue

        next_e = prev_e * row["yoy_e"]
        next_p = prev_p * row["yoy_p"]

        proj_rows.append({
            "year": year,
            "capex_e": round(next_e,1),
            "capex_p": round(next_p,1),
            "region": "World",
            "tech": "BESS",
            "variable": "capex",
            "units": "kWh",
            "type": "projected",
            "money": "USD",
            "money year": 2024,
            "source": "Derived from iterative Modo YoY"
        })

        prev_e = next_e
        prev_p = next_p

    return pd.DataFrame(proj_rows)


# -------------------------------------------------------------------
# Build long-format CAPEX-Opex output
# -------------------------------------------------------------------
def build_long_capex(world_hist: pd.DataFrame,
                     world_proj: pd.DataFrame,
                     world_raw: pd.DataFrame) -> pd.DataFrame:

    # combine CAPEX
    all_capex = pd.concat([world_hist, world_proj], ignore_index=True)

    rows = []

    for _, row in all_capex.iterrows():

        # energy CAPEX
        rows.append({
            "year": row["year"],
            "region": row.get("region", "World"),
            "tech": "bess_energy",
            "variable": "capex",
            "value": row["capex_e"],
            "units": "kWh",
            "type": row.get("type", "historic"),
            "money": row.get("money", "USD"),
            "money year": row.get("money year", 2024),
            "source": row.get("source", "Derived"),
        })

        # power CAPEX
        rows.append({
            "year": row["year"],
            "region": row.get("region", "World"),
            "tech": "bess_power",
            "variable": "capex",
            "value": row["capex_p"],
            "units": "kW",
            "type": row.get("type", "historic"),
            "money": row.get("money", "USD"),
            "money year": row.get("money year", 2024),
            "source": row.get("source", "Derived"),
        })

    # ------------------------------------------------------------------
    # Add ONE OPEX line (energy only)
    # ------------------------------------------------------------------
    opex_row_src = world_raw[
        (world_raw["variable"] == "opex_f") &
        (world_raw["tech"] == "BESS")
    ].iloc[0]

    rows.append({
        "year": "all",                     # matches your convention
        "region": opex_row_src["region"],
        "tech": "bess_energy",
        "variable": "opex_f",
        "value": opex_row_src["value"],
        "units": opex_row_src["units"],    # kWh/year
        "type": opex_row_src["type"],
        "money": opex_row_src["money"],
        "money year": opex_row_src["money year"],
        "source": opex_row_src["source"],
    })

    df_out = pd.DataFrame(rows)

    # ---------------------------------------------------
    # SORT BY variable → year before exporting
    # CAPEX rows sort numerically, OPEX remains last
    # ---------------------------------------------------
    df_out["year_sort"] = df_out["year"].replace("all", 9999).astype(int)
    df_out = df_out.sort_values(["variable", "year_sort"]).drop(columns=["year_sort"])

    return df_out



# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    world_raw = pd.read_csv(WORLD_FILE)
    modo_raw  = pd.read_csv(MODO_FILE)

    world_raw["year"] = world_raw["year"].replace("all", np.nan)
    world_raw["year"] = world_raw["year"].astype(float)

    world_hist_capex = split_world_capex_energy_power(world_raw)
    modo_yoy = compute_modo_decline_factors(modo_raw)
    world_proj_capex = project_world_from_modo(world_hist_capex, modo_yoy)

    # clean types
    world_hist_capex["year"] = world_hist_capex["year"].astype(int)
    world_proj_capex["year"] = world_proj_capex["year"].astype(int)

    # build final CAPEX-only output
    capex_long = build_long_capex(world_hist_capex, world_proj_capex, world_raw)
    capex_long = capex_long.sort_values(["tech", "year"], na_position="last")

    # save
    capex_long.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved CAPEX-only split to:\n  {OUTPUT_FILE}")

    print("\nSample output:")
    print(capex_long.head(10))


if __name__ == "__main__":
    main()
