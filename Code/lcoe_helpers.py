# lcoe helper functions
import logging
logger = logging.getLogger(__name__)

# lcoe_helpers.py
import logging
import pandas as pd
import numpy as np
from reader import get_val
from lcoe.lcoe import lcoe

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
        bess_capacity_mwh: float,
        availability: float,
        capex_opex_df: pd.DataFrame,
        discount_rate: float | None = None,
        lifetime: float | None = None,
        audit_log: list | None = None,     # kept in signature but unused
        result_id: int | None = None,      # same here
) -> dict:
    """
    Calculates discounted LCOE for a solar+BESS system with a simple breakdown:
        - Solar CAPEX contribution
        - BESS CAPEX contribution
        - Opex contribution

    Assumes:
        - All CAPEX is paid at t=0
        - Opex and energy occur once per year for `lifetime` years
        - First year's energy/opex at t=0 (to match npf.pv(..., when=1) convention)

    No financing / principal / interest logic.
    """

    # --- Inputs from table ---
    solar_capex = get_val(capex_opex_df, country, year, "capex", "solar",
                          audit_log=audit_log, audit_context=None)
    bess_capex = get_val(capex_opex_df, country, year, "capex", "bess",
                         audit_log=audit_log, audit_context=None)

    solar_opex = get_val(capex_opex_df, country, "all", "opex_f", "solar",
                         audit_log=audit_log, audit_context=None)
    bess_opex = get_val(capex_opex_df, country, "all", "opex_f", "bess",
                        audit_log=audit_log, audit_context=None)

    if discount_rate is None:
        discount_rate = get_val(capex_opex_df, country, "all", "wacc", "solar",
                                audit_log=audit_log, audit_context=None)

    if lifetime is None:
        lifetime = int(get_val(capex_opex_df, country, "all", "life", "solar",
                               audit_log=audit_log, audit_context=None))

    af = _to_frac(availability)
    r = _to_frac(discount_rate)

    # --- Costs (all in same currency) ---
    solar_capex_total = solar_capacity_mw * solar_capex * 1000
    bess_capex_total = bess_capacity_mwh * bess_capex * 1000
    total_capex = solar_capex_total + bess_capex_total

    annual_opex = (
        solar_capacity_mw * solar_opex * 1000 +
        bess_capacity_mwh * bess_opex * 1000
    )

    # --- Energy ---
    annual_energy_mwh = af * 8760

    # --- Discounting (first energy/opex at t=0: 0..lifetime-1) ---
    discount_factors = 1 / (1 + r) ** np.arange(0, lifetime)

    pv_energy = (annual_energy_mwh * discount_factors).sum()
    pv_opex = (annual_opex * discount_factors).sum()

    # --- Simple component LCOEs ---
    solar_capex_lcoe = solar_capex_total / pv_energy
    bess_capex_lcoe = bess_capex_total / pv_energy
    opex_lcoe = pv_opex / pv_energy

    # Total LCOE = PV(costs) / PV(energy)
    lcoe_val = solar_capex_lcoe + bess_capex_lcoe + opex_lcoe

    # --- Breakdown table ---
    breakdown = {
        "Solar CAPEX": solar_capex_lcoe,
        "BESS CAPEX": bess_capex_lcoe,
        "Opex": opex_lcoe,
    }

    breakdown_df = pd.DataFrame.from_dict(
        breakdown, orient="index", columns=["Value"]
    )
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
    audit_log: list | None = None,     # NEW
    result_id: int | None = None,      # NEW
) -> dict:
    """
    Calculates LCOE for a conventional power plant.
    Logs:
        - warnings on fallback
        - errors on missing inputs
    Records:
        - audit trail of input rows used
    """
    ctx = {
        "calc_type": f"conventional_{tech.lower()}",
        "calc_country": country,
        "calc_year": year,
        "calc_capacity_mw": capacity_mw,
        "calc_capacity_factor": capacity_factor,
        "calc_result_id": result_id,
    }

    try:
        capex_kw = get_val(
            capex_opex_df, country, "all", "capex", tech,
            audit_log=audit_log, audit_context=ctx
        )
        opex_fixed_kwyr = get_val(
            capex_opex_df, country, "all", "opex_f", tech,
            audit_log=audit_log, audit_context=ctx
        )
        opex_var_mwh = get_val(
            capex_opex_df, country, "all", "opex_v", tech,
            audit_log=audit_log, audit_context=ctx
        )
        fuel_mwh = get_val(
            capex_opex_df, country, year, "fuel", tech,
            audit_log=audit_log, audit_context=ctx
        )
        efficiency = _to_frac(get_val(
            capex_opex_df, country, "all", "efficiency", tech,
            audit_log=audit_log, audit_context=ctx
        ))

        if discount_rate is None:
            discount_rate = get_val(
                capex_opex_df, country, "all", "wacc", "solar",
                audit_log=audit_log, audit_context=ctx
            )

        if lifetime is None:
            lifetime = int(get_val(
                capex_opex_df, country, "all", "life", tech,
                audit_log=audit_log, audit_context=ctx
            ))

        cf = _to_frac(capacity_factor)
        r = _to_frac(discount_rate)

        total_capex = capacity_mw * 1000 * capex_kw
        annual_fixed_opex = capacity_mw * 1000 * opex_fixed_kwyr

        annual_energy_mwh = capacity_mw * 8760 * cf
        fuel_cost_mwh_elec = fuel_mwh / efficiency
        variable_cost_mwh = opex_var_mwh + fuel_cost_mwh_elec
        annual_variable_cost = variable_cost_mwh * annual_energy_mwh

        annual_opex_total = annual_fixed_opex + annual_variable_cost

        lcoe_val = lcoe(
            annual_energy_mwh, total_capex,
            annual_opex_total, r, lifetime
        )

        return {"LCOE": lcoe_val, "Total_Capex": total_capex}

    except ValueError as e:
        logger.error(
            "Could not calculate %s LCOE for %s (%s): %s",
            tech, country, year, e,
        )
        # error is re-raised so main workflow can skip this tech/year
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