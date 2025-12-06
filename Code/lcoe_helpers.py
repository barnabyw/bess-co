# lcoe helper functions
import logging
logger = logging.getLogger(__name__)

# lcoe_helpers.py
import logging
import pandas as pd
import numpy as np
from Code.data_prep.reader import get_val
from lcoe.lcoe import lcoe
from augmentation import optimise_augmentation

logger = logging.getLogger(__name__)

def _to_frac(x):
    """Allow 0–1 or 0–100 inputs; return fraction 0–1."""
    x = float(x)
    return x / 100.0 if x > 1 else x

# ---------------------------------------------------------------------------
#   SOLAR + BESS LCOE WITH AUDIT + LOGGING
# ---------------------------------------------------------------------------
def calculate_solar_bess_lcoe(
    country: str,
    year: int,
    solar_capacity_mw: float,
    bess_energy_mwh: float,
    bess_power_mw: float,
    availability: float,
    bess_cycles: float,
    capex_opex_df: pd.DataFrame,
    scenario: str | None = None,     # NEW
    discount_rate: float | None = None,
    lifetime: float | None = None,
    result_id: int | None = None,
):
    """
    Calculates discounted LCOE for a solar+BESS system with separate
    BESS power and energy CAPEX components.

    New:
        - BESS CAPEX = power_capex * bess_power_mw + energy_capex * bess_energy_mwh
    """

    # ------------- CAPEX INPUTS -------------
    solar_capex = get_val(capex_opex_df, country, year, "capex", "solar")
    bess_energy_capex = get_val(capex_opex_df, country, year, "capex_e", "BESS")
    bess_power_capex = get_val(capex_opex_df, country, year, "capex_p", "BESS")

    # ------------- OPEX INPUTS -------------
    solar_opex = get_val(capex_opex_df, country, "all", "opex_f", "solar")
    bess_energy_opex = get_val(capex_opex_df, country, "all", "opex_f", "BESS")

    # ----------- Discount rate / lifetime ----------
    if discount_rate is None:
        discount_rate = get_val(capex_opex_df, country, "all", "wacc", "solar")

    if lifetime is None:
        lifetime = int(get_val(capex_opex_df, country, "all", "life", "solar"))

    af = _to_frac(availability)
    r = _to_frac(discount_rate)

    # ------------- CAPEX CALCULATION -------------
    solar_capex_total = solar_capacity_mw * solar_capex * 1000

    bess_energy_capex_total = bess_energy_mwh * bess_energy_capex * 1000
    bess_power_capex_total  = bess_power_mw   * bess_power_capex  * 1000

    total_capex = solar_capex_total + bess_energy_capex_total + bess_power_capex_total

    # ------------- ANNUAL OPEX -------------
    annual_opex = solar_capacity_mw  * solar_opex * 1000 + bess_energy_mwh * bess_energy_opex  * 1000

    # ------------- ENERGY PRODUCTION -------------
    annual_energy_mwh = af * 8760

    # ------------- DISCOUNT FACTORS -------------
    discount_factors = 1 / (1 + r) ** np.arange(0, lifetime)

    pv_energy = (annual_energy_mwh * discount_factors).sum()
    pv_opex = (annual_opex * discount_factors).sum()

    # ------------- AUGMENTATION MODEL -------------
    best_aug, _ = optimise_augmentation(
        optimal_bess_mwh=bess_energy_mwh,
        cycles_per_annum=bess_cycles,
        discount_rate=discount_rate,
        capex_opex_df=capex_opex_df,
        build_year=year,
        project_life=lifetime,
        project_energy_gwh_per_annum=annual_energy_mwh / 1000,
        scenario=scenario,
    )

    augmentation_disc = (
        best_aug["initial_capex_disc"] +
        best_aug["augmentation_capex_disc"]
    )

    # ------------- COMPONENT LCOEs -------------
    solar_capex_lcoe = solar_capex_total / pv_energy
    bess_energy_lcoe = bess_energy_capex_total / pv_energy
    bess_power_lcoe  = bess_power_capex_total  / pv_energy
    augmentation_lcoe = augmentation_disc / pv_energy
    opex_lcoe = pv_opex / pv_energy

    # GRAND TOTAL
    lcoe_val = (solar_capex_lcoe + bess_energy_lcoe + bess_power_lcoe +
                augmentation_lcoe + opex_lcoe)

    breakdown = {
        "Solar CAPEX": solar_capex_lcoe,
        "BESS Energy CAPEX": bess_energy_lcoe,
        "BESS Power CAPEX": bess_power_lcoe,
        "Augmentation": augmentation_lcoe,
        "Opex": opex_lcoe,
    }

    breakdown_df = pd.DataFrame.from_dict(breakdown, orient="index", columns=["Value"])
    breakdown_df.loc["Total"] = breakdown_df["Value"].sum()

    return {
        "LCOE": lcoe_val,
        "Total_Capex": total_capex,
        "Breakdown": breakdown_df,
    }

# ---------------------------------------------------------------------------
#   CONVENTIONAL LCOE WITH AUDIT + LOGGING
# ---------------------------------------------------------------------------
def calculate_conventional_lcoe(
    country: str,
    year: int,
    tech: str,
    capacity_mw: float,
    capacity_factor: float,
    capex_opex_df: pd.DataFrame,
    discount_rate: float | None = None,
    lifetime: float | None = None,
    audit_log: list | None = None,
    result_id: int | None = None,
) -> dict:
    """
    Calculates LCOE for a conventional power plant (Coal, Gas).

    Breakdown categories (simplified):
        - CAPEX
        - Opex (fixed + variable excluding fuel)
        - Fuel
    """

    try:
        # ---- INPUTS ----
        capex_kw           = get_val(capex_opex_df, country, "all", "capex", tech)
        opex_fixed_kwyr    = get_val(capex_opex_df, country, "all", "opex_f", tech)
        opex_var_mwh       = get_val(capex_opex_df, country, "all", "opex_v", tech)
        fuel_mwh           = get_val(capex_opex_df, country, year, "fuel", tech)
        efficiency         = _to_frac(get_val(capex_opex_df, country, "all", "efficiency", tech))

        # fallback
        if discount_rate is None:
            discount_rate = get_val(capex_opex_df, country, "all", "wacc", "solar")

        if lifetime is None:
            lifetime = int(get_val(capex_opex_df, country, "all", "life", tech))

        cf = _to_frac(capacity_factor)
        r  = _to_frac(discount_rate)

        # ---- CAPEX ----
        total_capex = capacity_mw * 1000 * capex_kw  # $ total

        # ---- ENERGY ----
        annual_energy_mwh = capacity_mw * 8760 * cf

        # ---- OPEX ----
        annual_fixed_opex    = capacity_mw * 1000 * opex_fixed_kwyr
        annual_variable_opex = opex_var_mwh * annual_energy_mwh

        # ---- FUEL ----
        fuel_cost_mwh_elec = fuel_mwh / efficiency
        annual_fuel_cost   = fuel_cost_mwh_elec * annual_energy_mwh

        # ---- DISCOUNT FACTORS ----
        discount_factors = 1 / (1 + r) ** np.arange(0, lifetime)

        pv_energy = (annual_energy_mwh * discount_factors).sum()

        pv_capex   = total_capex
        pv_opex    = (annual_fixed_opex * discount_factors).sum() \
                     + (annual_variable_opex * discount_factors).sum()
        pv_fuel    = (annual_fuel_cost   * discount_factors).sum()

        # ---- COMPONENT LCOE ----
        capex_lcoe = pv_capex / pv_energy
        opex_lcoe  = pv_opex  / pv_energy
        fuel_lcoe  = pv_fuel  / pv_energy

        total_lcoe = capex_lcoe + opex_lcoe + fuel_lcoe

        # ---- BREAKDOWN ----
        breakdown = {
            "CAPEX": capex_lcoe,
            "Opex": opex_lcoe,
            "Fuel": fuel_lcoe,
        }

        breakdown_df = pd.DataFrame.from_dict(breakdown, orient="index", columns=["Value"])
        breakdown_df.loc["Total"] = breakdown_df["Value"].sum()

        return {
            "LCOE": total_lcoe,
            "Total_Capex": total_capex,
            "Breakdown": breakdown_df,
        }

    except ValueError as e:
        logger.error(
            f"Could not calculate {tech} LCOE for {country} ({year}): {e}"
        )
        raise

if __name__ == "__main__":
    import os
    from logging_conf import setup_logging

    # === Configuration ===
    CWD = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(CWD, "..", "inputs")
    OUTPUT_PATH = os.path.join(CWD, "..", "outputs")

    BASE_YEAR = 2024
    CONVENTIONAL_TECHS = ["Coal", "Gas"]
    APPEND_RESULTS = True

    # === Logging ===
    setup_logging(OUTPUT_PATH)
    logger = logging.getLogger("main_workflow")

    # === Global audit store ===
    AUDIT_LOG = []

    # === Load Data ===
    logger.info("Loading input data...")
    countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))
    capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))

    solar_cap = 3.9
    bess_energy = 13.6
    avail = 0.75
    country = "United States"

    sb_result = calculate_solar_bess_lcoe(
        country, BASE_YEAR, solar_cap, bess_energy, avail, capex_opex_df)

    print(sb_result)