from assumptions import *

df = pd.read_csv(os.path.join(output_path, "results.csv"))

import pandas as pd
import numpy as np
from lcoe.lcoe import lcoe


def calculate_multi_year_costs_lcoe(results_2025_df, capex_learning_df):
    """
    Calculate costs and LCOE for multiple years using fixed 2025 capacities
    but varying technology costs.

    Parameters:
    - results_2025_df: DataFrame with 2025 optimization results
    - cost_projection_df: DataFrame with cost projections by year

    Returns:
    - DataFrame with results for all years
    """

    # Create a list to store results for all years
    all_results = []

    # Extract the base columns that don't change
    base_columns = ['Country', 'Latitude', 'Longitude', 'Solar_Capacity', 'BESS_Energy']
    base_data = results_2025_df[base_columns].copy()

    # For each year in the cost projection
    for _, cost_row in capex_learning_df.iterrows():
        year = cost_row['year']
        solar_cost_per_mw = cost_row['solar_cost_per_mw']
        bess_cost_per_mwh = cost_row['bess_energy_cost_per_mwh']

        # Create a copy of base data for this year
        year_data = base_data.copy()
        year_data['Year'] = year

        # Calculate new costs based on capacities and new cost parameters
        cost = year_data['Solar_Capacity'] * solar_cost_per_mw + year_data['BESS_Energy'] * bess_cost_per_mwh
        year_data['Cost'] = cost

        year_data['LCOE'] = 1000 * lcoe(load * 8760 * target, cost, 0, 0.08, 20)

        all_results.append(year_data)

    # Combine all years into a single DataFrame
    final_results = pd.concat(all_results, ignore_index=True)

    # Reorder columns to match original format
    column_order = ['Country', 'Latitude', 'Longitude', 'LCOE', 'Cost', 'Year',
                    'Solar_Capacity', 'BESS_Energy']
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
