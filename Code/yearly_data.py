from assumptions import *
from reader import get_val

df = pd.read_csv(os.path.join(output_path, "results.csv"))

import pandas as pd
import numpy as np
from lcoe.lcoe import lcoe


def calculate_multi_year_costs_lcoe(results_year, capex_df, years):
    """
    Calculate costs and LCOE for multiple years using fixed 2025 capacities
    but varying technology costs.

    Parameters:
    - results_year: DataFrame with 2025 optimization results
    - cost_projection_df: DataFrame with cost projections by year

    Returns:
    - DataFrame with results for all years
    """

    # Create a list to store results for all years
    all_results = []

    # For each year in the cost projection
    for _, row in results_year.iterrows():
        year = row['Year']
        solar_cap = row['Solar_Capacity']
        bess_cap = row['BESS_Energy']

        country = row["Country"]
        lat = row["Latitude"]
        lon = row["Longitude"]

        for year in years:
            solar_capex = get_val(capex_df, country, year, "capex")
            bess_capex = get_val(capex_df, country, year, "capex")

        cost = solar_cap * solar_capex + bess_cap * bess_capex

        lcoe_val = 1000 * lcoe(load * 8760 * target, cost, 0, 0.08, 20)
        # Store or print results as needed
        all_results.append({
            "Country": country,
            "Latitude": lat,
            "Longitude": lon,
            "LCOE": lcoe_val,
            "Cost": cost,
            "Year": year,
            "Solar_Capacity": solar_cap,
            "BESS_Energy": bess_cap,
            "Tech": "Solar BESS"
        })

    # Combine all years into a single DataFrame
    final_results = pd.concat(all_results, ignore_index=True)

    # Reorder columns to match original format
    column_order = ['Country', 'Latitude', 'Longitude', 'LCOE', 'Cost', 'Year',
                    'Solar_Capacity', 'BESS_Energy', "Tech"]
    final_results = final_results[column_order]

    return final_results


# Example usage:
if __name__ == "__main__":
    # Load your data
    # results_2025 = pd.read_csv('your_2025_results.csv')
    # cost_projections = pd.read_csv('your_cost_projections.csv')

    # Example with sample data
    results_2025 = pd.read_csv(os.path.join(output_path, "results.csv"))

    # Calculate results
    results = calculate_multi_year_costs_lcoe(results_2025, capex_learning_df)

    print("Sample results:")
    print(results.head(10))

    # Save results
    results.to_csv(os.path.join(output_path, "multi_yearly_results_80.csv"), index=False)

    print(f"\nTotal rows: {len(results)}")
    print(f"Countries: {results['Country'].nunique()}")
    print(f"Years: {sorted(results['Year'].unique())}")
