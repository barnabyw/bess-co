# lcoe helper functions
import pandas as pd
from reader import get_val
from lcoe.lcoe import lcoe

def _to_frac(x):
    """Allow 0–1 or 0–100 inputs; return fraction 0–1."""
    x = float(x)
    return x/100.0 if x > 1 else x

def calculate_solar_bess_lcoe(
        country: str,
        year: int,
        solar_capacity_mw: float,
        bess_capacity_mwh: float,
        availability: float,
        capex_opex_df: pd.DataFrame
) -> dict:
    """Calculates LCOE for a fixed capacity solar+BESS system for a given year."""

    # This is a simplified LCOE calculation.
    # You should replace this with your detailed lcoe.lcoe function.
    try:
        # Get all required financial and technical parameters for the given year
        solar_capex = get_val(capex_opex_df, country, year, "capex", "Solar")
        bess_capex = get_val(capex_opex_df, country, year, "capex", "BESS")
        solar_opex = get_val(capex_opex_df, country, year, "opex", "Solar", param_type="fixed")
        bess_opex = get_val(capex_opex_df, country, year, "opex", "BESS", param_type="fixed")
        discount_rate = get_val(capex_opex_df, country, year, "discount_rate")  # Assumes discount_rate is a variable
        solar_lifetime = int(get_val(capex_opex_df, country, year, "lifetime", "Solar"))

        af = _to_frac(availability)
        r = _to_frac(discount_rate)

        # 1. Total Investment Cost
        total_capex = (solar_capacity_mw * solar_capex * 1000) + (bess_capacity_mwh * bess_capex * 1000)

        # 2. Total Annual Costs
        annual_opex = (solar_capacity_mw * solar_opex * 1000) + (bess_capacity_mwh * bess_opex * 1000)

        # Use the optimised result from 'optimise_bess' function
        annual_energy_mwh = af * 8760

        # lcoe = total_annual_cost / (annual_energy_gwh * 1000) # in USD/MWh
        lcoe_val = lcoe(annual_energy_mwh, total_capex, annual_opex, r, solar_lifetime)

        return {"LCOE": lcoe_val, "Total_Capex": total_capex}

    except ValueError as e:
        print(f"  - Could not calculate Solar+BESS LCOE for {year}: {e}")
        return None

def calculate_conventional_lcoe(
    country: str,
    year: int,
    tech: str,
    capacity_mw: float,
    capacity_factor: float,
    capex_opex_df: pd.DataFrame
) -> dict:
    """Calculates LCOE for a conventional power plant for a given year (simple version)."""
    try:
        # Inputs from table
        capex_kw = get_val(capex_opex_df, country, year, "capex", tech)                          # $/kW
        opex_fixed_kwyr = get_val(capex_opex_df, country, year, "opex", tech, param_type="fixed") # $/kW/yr
        opex_var_mwh = get_val(capex_opex_df, country, year, "opex", tech, param_type="variable") # $/MWh
        fuel_cost_mwh_fuel = get_val(capex_opex_df, country, year, "fuel", tech)                  # $/MWh_fuel
        efficiency = _to_frac(get_val(capex_opex_df, country, year, "efficiency", tech))          # 0–1 (or 0–100)
        discount_rate = get_val(capex_opex_df, country, year, "discount_rate")                    # 0–1 (or 0–100)
        lifetime = int(get_val(capex_opex_df, country, year, "lifetime", tech))                   # years

        cf = _to_frac(capacity_factor)
        r = _to_frac(discount_rate)

        # Derived quantities
        total_capex = capacity_mw * 1000 * capex_kw  # $
        annual_fixed_opex = capacity_mw * 1000 * opex_fixed_kwyr  # $/yr

        # Energy and variable costs
        annual_energy_mwh = capacity_mw * 8760 * cf  # MWh/yr
        fuel_per_mwh_elec = fuel_cost_mwh_fuel / efficiency  # $/MWh_e
        variable_cost_per_mwh = opex_var_mwh + fuel_per_mwh_elec  # $/MWh_e
        annual_variable_cost = variable_cost_per_mwh * annual_energy_mwh  # $/yr

        # Feed into your existing lcoe() function
        annual_opex_total = annual_fixed_opex + annual_variable_cost  # $/yr
        lcoe_val = lcoe(annual_energy_mwh, total_capex, annual_opex_total, r, lifetime)

        return {"LCOE": lcoe_val, "Total_Capex": total_capex}

    except ValueError as e:
        print(f" - Could not calculate {tech} LCOE for {year}: {e}")
        return None


    except ValueError as e:
        print(f"  - Could not calculate {tech} LCOE for {year}: {e}")
        return None