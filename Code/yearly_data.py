from assumptions import *

df = pd.read_csv(os.path.join(output_path, "results.csv"))

import pandas as pd
import numpy as np


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
        year_data['Cost'] = (
                year_data['Solar_Capacity'] * solar_cost_per_mw +
                year_data['BESS_Energy'] * bess_cost_per_mwh
        )

        # Calculate LCOE (assuming same methodology as original)
        # You may need to adjust this based on your specific LCOE calculation
        # This assumes LCOE scales proportionally with cost changes
        cost_ratio = year_data['Cost'] / results_2025_df['Cost']
        year_data['LCOE'] = results_2025_df['LCOE'] * cost_ratio

        all_results.append(year_data)

    # Combine all years into a single DataFrame
    final_results = pd.concat(all_results, ignore_index=True)

    # Reorder columns to match original format
    column_order = ['Country', 'Latitude', 'Longitude', 'LCOE', 'Cost', 'Year',
                    'Solar_Capacity', 'BESS_Energy']
    final_results = final_results[column_order]

    return final_results


# Alternative vectorized approach for better performance
def calculate_multi_year_costs_lcoe_vectorized(results_2025_df, cost_projection_df):
    """
    Vectorized version for better performance with large datasets.
    """

    # Create base data without year-specific columns
    base_columns = ['Country', 'Latitude', 'Longitude', 'Solar_Capacity', 'BESS_Energy']
    base_data = results_2025_df[base_columns].copy()

    # Create cartesian product of countries × years
    countries = base_data
    years = cost_projection_df

    # Use merge to create all combinations
    final_results = countries.assign(key=1).merge(
        years.assign(key=1), on='key'
    ).drop('key', axis=1)

    # Calculate costs vectorized
    final_results['Cost'] = (
            final_results['Solar_Capacity'] * final_results['solar_cost_per_mw'] +
            final_results['BESS_Energy'] * final_results['bess_energy_cost_per_mwh']
    )

    # Calculate LCOE using broadcast operations
    # Merge with 2025 results to get original LCOE and Cost for ratio calculation
    results_2025_subset = results_2025_df[['Country', 'LCOE', 'Cost']].copy()
    results_2025_subset.columns = ['Country', 'LCOE_2025', 'Cost_2025']

    final_results = final_results.merge(results_2025_subset, on='Country')

    # Calculate new LCOE based on cost ratio
    final_results['LCOE'] = (
            final_results['LCOE_2025'] *
            (final_results['Cost'] / final_results['Cost_2025'])
    )

    # Clean up temporary columns
    final_results = final_results.drop(['solar_cost_per_mw', 'bess_energy_cost_per_mwh',
                                        'LCOE_2025', 'Cost_2025'], axis=1)

    # Reorder columns
    column_order = ['Country', 'Latitude', 'Longitude', 'LCOE', 'Cost', 'year',
                    'Solar_Capacity', 'BESS_Energy']
    final_results = final_results[column_order]
    final_results.rename(columns={'year': 'Year'}, inplace=True)

    return final_results.sort_values(['Country', 'Year']).reset_index(drop=True)


# Example usage:
if __name__ == "__main__":
    # Load your data
    # results_2025 = pd.read_csv('your_2025_results.csv')
    # cost_projections = pd.read_csv('your_cost_projections.csv')

    # Example with sample data
    results_2025 = pd.read_csv(os.path.join(output_path, "results.csv"))

    # Calculate results using vectorized approach (recommended for large datasets)
    results = calculate_multi_year_costs_lcoe_vectorized(
        results_2025, capex_learning_df
    )

    print("Sample results:")
    print(results.head(10))

    # Save results
    results.to_csv(os.path.join(output_path, "multi_yearly_results.csv"), index=False)

    print(f"\nTotal rows: {len(results)}")
    print(f"Countries: {results['Country'].nunique()}")
    print(f"Years: {sorted(results['Year'].unique())}")
