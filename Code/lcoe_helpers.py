# lcoe_helpers.py
import pandas as pd
import numpy as np
from Code.data_prep.reader import get_val
from augmentation import optimise_augmentation

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _to_frac(x):
    """Allow 0–1 or 0–100 inputs; return fraction 0–1."""
    x = float(x)
    return x / 100.0 if x > 1 else x

# ------------------------------------------------------------
# SOLAR + BESS LCOE
# ------------------------------------------------------------
def calculate_solar_bess_lcoe(
    country: str,
    year: int,
    solar_capacity_mw: float,
    bess_energy_mwh: float,
    bess_power_mw: float,
    availability: float,
    bess_cycles: float,
    capex_opex_df: pd.DataFrame,
    scenario: str | None = None,
    discount_rate: float | None = None,
    lifetime: float | None = None,
    result_id: int | None = None,
    used_fallbacks: dict | None = None,   # <---- NEW
):
    # ------------------------------------------------------------
    # INPUT PARAMETERS
    # ------------------------------------------------------------
    solar_capex = get_val(capex_opex_df, country, year, "capex", "solar",
                          used_fallbacks=used_fallbacks)
    bess_energy_capex = get_val(capex_opex_df, country, year, "capex_e", "BESS",
                                used_fallbacks=used_fallbacks)
    bess_power_capex  = get_val(capex_opex_df, country, year, "capex_p", "BESS",
                                used_fallbacks=used_fallbacks)

    solar_opex = get_val(capex_opex_df, country, "all", "opex_f", "solar",
                         used_fallbacks=used_fallbacks)
    bess_power_opex = get_val(capex_opex_df, country, "all", "opex_f", "BESS",
                              used_fallbacks=used_fallbacks)
    bess_variable_opex = get_val(capex_opex_df, country, "all", "opex_v", "BESS",
                                 used_fallbacks=used_fallbacks)

    if discount_rate is None:
        discount_rate = get_val(capex_opex_df, country, "all", "wacc", "solar",
                                used_fallbacks=used_fallbacks)
    if lifetime is None:
        lifetime = int(get_val(capex_opex_df, country, "all", "life", "solar",
                               used_fallbacks=used_fallbacks))

    af = _to_frac(availability)
    r = _to_frac(discount_rate)

    # ------------------------------------------------------------
    # BASE CAPEX
    # ------------------------------------------------------------
    solar_capex_total = solar_capacity_mw * solar_capex * 1000
    bess_energy_capex_total = bess_energy_mwh * bess_energy_capex * 1000
    bess_power_capex_total  = bess_power_mw  * bess_power_capex  * 1000

    # ------------------------------------------------------------
    # OPEX + ENERGY
    # ------------------------------------------------------------
    annual_energy_mwh = af * 8760
    discount_factors = 1 / (1 + r) ** np.arange(0, lifetime)

    annual_opex = (
        solar_capacity_mw * solar_opex * 1000 +
        bess_power_mw * bess_power_opex * 1000 +
        annual_energy_mwh * bess_variable_opex
    )

    pv_energy = (annual_energy_mwh * discount_factors).sum()
    pv_opex = (annual_opex * discount_factors).sum()

    # ------------------------------------------------------------
    # AUGMENTATION MODEL
    # ------------------------------------------------------------
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

    # Initial oversizing (full cost paid in year 0 → no discounting)
    extra_initial_mwh = max(0, best_aug["initial_capacity_mwh"] - bess_energy_mwh)
    initial_oversize_cost = extra_initial_mwh * bess_energy_capex * 1000

    # Mid-life augmentation (already discounted inside optimiser)
    augmentation_disc = float(best_aug.get("augmentation_capex_disc", 0.0))

    # Corrected BESS energy CAPEX
    bess_energy_capex_total_corrected = (
        bess_energy_capex_total + initial_oversize_cost
    )

    # ------------------------------------------------------------
    # TOTAL CAPEX
    # ------------------------------------------------------------
    total_capex = (
        solar_capex_total +
        bess_power_capex_total +
        bess_energy_capex_total_corrected +
        augmentation_disc
    )

    # ------------------------------------------------------------
    # LCOE COMPONENTS
    # ------------------------------------------------------------
    solar_capex_lcoe  = solar_capex_total / pv_energy
    bess_energy_lcoe  = bess_energy_capex_total_corrected / pv_energy
    bess_power_lcoe   = bess_power_capex_total / pv_energy
    augmentation_lcoe = augmentation_disc / pv_energy
    opex_lcoe         = pv_opex / pv_energy

    total_lcoe = (
        solar_capex_lcoe +
        bess_energy_lcoe +
        bess_power_lcoe +
        augmentation_lcoe +
        opex_lcoe
    )

    # ------------------------------------------------------------
    # BREAKDOWN TABLE
    # ------------------------------------------------------------
    breakdown = {
        "Solar CAPEX": solar_capex_lcoe,
        "BESS Energy CAPEX": bess_energy_lcoe,
        "BESS Power CAPEX": bess_power_lcoe,
        "Augmentation": augmentation_lcoe,
        "Opex": opex_lcoe,
    }

    breakdown_df = pd.DataFrame.from_dict(breakdown, orient="index", columns=["Value"])
    breakdown_df.loc["Total"] = total_lcoe

    return {
        "LCOE": total_lcoe,
        "Total_Capex": total_capex,
        "Breakdown": breakdown_df,
    }

# ------------------------------------------------------------
# CONVENTIONAL LCOE
# ------------------------------------------------------------
def calculate_conventional_lcoe(
    country: str,
    year: int,
    tech: str,
    capacity_mw: float,
    capacity_factor: float,
    capex_opex_df: pd.DataFrame,
    discount_rate: float | None = None,
    lifetime: float | None = None,
    result_id: int | None = None,
    used_fallbacks: dict | None = None,   # <---- NEW
) -> dict:

    # === INPUT PARAMETER LOOKUPS (all audited now) ===
    capex_kw        = get_val(capex_opex_df, country, "all", "capex", tech,
                              used_fallbacks=used_fallbacks)
    opex_fixed_kwyr = get_val(capex_opex_df, country, "all", "opex_f", tech,
                              used_fallbacks=used_fallbacks)
    opex_var_mwh    = get_val(capex_opex_df, country, "all", "opex_v", tech,
                              used_fallbacks=used_fallbacks)
    fuel_mwh        = get_val(capex_opex_df, country, year, "fuel", tech,
                              used_fallbacks=used_fallbacks)
    efficiency      = _to_frac(get_val(capex_opex_df, country, "all", "efficiency", tech,
                                      used_fallbacks=used_fallbacks))

    if discount_rate is None:
        discount_rate = get_val(capex_opex_df, country, "all", "wacc", "solar",
                                used_fallbacks=used_fallbacks)
    if lifetime is None:
        lifetime = int(get_val(capex_opex_df, country, "all", "life", tech,
                               used_fallbacks=used_fallbacks))

    # === BASIC TERMS ===
    cf = _to_frac(capacity_factor)
    r  = _to_frac(discount_rate)

    # CAPEX (MW → kW)
    total_capex = capacity_mw * 1000 * capex_kw

    # ENERGY
    annual_energy_mwh = capacity_mw * 8760 * cf

    # OPEX
    annual_fixed_opex = capacity_mw * 1000 * opex_fixed_kwyr
    annual_variable_opex = opex_var_mwh * annual_energy_mwh

    # FUEL (cost per MWh electricity generated)
    fuel_cost_mwh_elec = fuel_mwh / efficiency
    annual_fuel_cost = fuel_cost_mwh_elec * annual_energy_mwh

    # DISCOUNTING
    discount_factors = 1 / (1 + r) ** np.arange(0, lifetime)

    pv_energy = (annual_energy_mwh * discount_factors).sum()
    pv_opex = (
        (annual_fixed_opex * discount_factors).sum() +
        (annual_variable_opex * discount_factors).sum()
    )
    pv_fuel = (annual_fuel_cost * discount_factors).sum()

    # LCOE COMPONENTS
    capex_lcoe = total_capex / pv_energy
    opex_lcoe  = pv_opex / pv_energy
    fuel_lcoe  = pv_fuel / pv_energy

    # BREAKDOWN TABLE
    breakdown = {
        "CAPEX": capex_lcoe,
        "Opex": opex_lcoe,
        "Fuel": fuel_lcoe,
    }

    breakdown_df = pd.DataFrame.from_dict(breakdown, orient="index", columns=["Value"])
    breakdown_df.loc["Total"] = breakdown_df["Value"].sum()

    # === FINAL OUTPUT ===
    return {
        "LCOE": capex_lcoe + opex_lcoe + fuel_lcoe,
        "Total_Capex": total_capex,
        "Breakdown": breakdown_df,
    }
