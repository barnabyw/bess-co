# lcoe helper functions
import pandas as pd
from reader import get_val
from lcoe.lcoe import lcoe

def calculate_solar_bess_lcoe(
        country: str,
        year: int,
        solar_capacity_mw: float,
        bess_capacity_mwh: float,
        solar_profile_mwh: pd.Series,
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
        solar_lifetime = get_val(capex_opex_df, country, year, "lifetime", "Solar")

        # 1. Total Investment Cost
        total_capex = (solar_capacity_mw * solar_capex * 1000) + (bess_capacity_mwh * bess_capex * 1000)

        # 2. Total Annual Costs
        annual_opex = (solar_capacity_mw * solar_opex * 1000) + (bess_capacity_mwh * bess_opex * 1000)

        # Let's use the optimized result from your 'optimise_bess' function as a proxy.
        annual_energy_gwh = solar_profile_mwh.sum() / 1000  # Example, should be dispatched energy

        # A placeholder for your more complex LCOE calculation
        # lcoe = total_annual_cost / (annual_energy_gwh * 1000) # in USD/MWh
        lcoe_val = lcoe(total_capex * 0.1) / (annual_energy_gwh * 1000)  # Simplified placeholder

        return {"LCOE": lcoe, "Total_Capex": total_capex}

    except ValueError as e:
        print(f"  - Could not calculate Solar+BESS LCOE for {year}: {e}")
        return None


def calculate_conventional_lcoe(
        country: str,
        year: int,
        tech: str,
        capex_opex_df: pd.DataFrame
) -> dict:
    """Calculates LCOE for a conventional power plant for a given year."""

    # This is a simplified LCOE calculation.
    # You should replace this with your detailed lcoe.lcoe function.
    try:
        capex = get_val(capex_opex_df, country, year, "capex", tech)
        opex_fixed = get_val(capex_opex_df, country, year, "opex", tech, param_type="fixed")
        fuel_cost = get_val(capex_opex_df, country, year, "fuel", tech)
        efficiency = get_val(capex_opex_df, country, year, "efficiency", tech)
        capacity_factor = get_val(capex_opex_df, country, year, "capacity_factor", tech)
        discount_rate = get_val(capex_opex_df, country, year, "discount_rate")
        lifetime = get_val(capex_opex_df, country, year, "lifetime", tech)

        # A placeholder for your LCOE calculation
        # lcoe = ... your formula here ...
        lcoe = (capex * 100) + (fuel_cost * 3) + (opex_fixed * 10)  # Simplified placeholder

        return {"LCOE": lcoe}

    except ValueError as e:
        print(f"  - Could not calculate {tech} LCOE for {year}: {e}")
        return None