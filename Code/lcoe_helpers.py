# lcoe helper functions
import pandas as pd
from reader import get_val
from lcoe.lcoe import lcoe
import logging
logger = logging.getLogger(__name__)

def _to_frac(x):
    """Allow 0–1 or 0–100 inputs; return fraction 0–1."""
    x = float(x)
    return x/100.0 if x > 1 else x

# lcoe_helpers.py
import logging
import pandas as pd
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
        audit_log: list | None = None,     # NEW
        result_id: int | None = None,      # NEW
) -> dict:
    """
    Calculates LCOE for a solar+BESS system.
    Includes:
        - full logging
        - audit logging of all input rows used
    """
    try:
        ctx = {
            "calc_type": "solar_bess",
            "calc_country": country,
            "calc_year": year,
            "calc_availability": availability,
            "calc_result_id": result_id,
        }

        solar_capex = get_val(
            capex_opex_df, country, year, "capex", "solar",
            audit_log=audit_log, audit_context=ctx
        )
        bess_capex = get_val(
            capex_opex_df, country, year, "capex", "bess",
            audit_log=audit_log, audit_context=ctx
        )

        solar_opex = get_val(
            capex_opex_df, country, "all", "opex_f", "solar",
            audit_log=audit_log, audit_context=ctx
        )
        bess_opex = get_val(
            capex_opex_df, country, "all", "opex_f", "bess",
            audit_log=audit_log, audit_context=ctx
        )

        discount_rate = get_val(
            capex_opex_df, country, "all", "wacc", "solar",
            audit_log=audit_log, audit_context=ctx
        )
        solar_lifetime = int(get_val(
            capex_opex_df, country, "all", "life", "solar",
            audit_log=audit_log, audit_context=ctx
        ))

        af = _to_frac(availability)
        r = _to_frac(discount_rate)

        # Investment cost
        total_capex = (
            solar_capacity_mw * solar_capex * 1000
            + bess_capacity_mwh * bess_capex * 1000
        )

        # Annual opex
        annual_opex = (
            solar_capacity_mw * solar_opex * 1000
            + bess_capacity_mwh * bess_opex * 1000
        )

        # Energy produced from optimised result
        annual_energy_mwh = af * 8760

        lcoe_val = lcoe(
            annual_energy_mwh, total_capex, annual_opex, r, solar_lifetime
        )

        return {"LCOE": lcoe_val, "Total_Capex": total_capex}

    except ValueError as e:
        logger.error(
            "Could not calculate Solar+BESS LCOE for %s (%s): %s",
            country, year, e,
        )
        return None


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
        discount_rate = get_val(
            capex_opex_df, country, "all", "wacc", "solar",
            audit_log=audit_log, audit_context=ctx
        )
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
