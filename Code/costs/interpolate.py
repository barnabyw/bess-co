import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_data():
    """Parse the raw data and create a structured DataFrame"""
    try:
        # Read Excel file
        data = pd.read_excel(r'C:\Users\barna\OneDrive\Documents\Solar_BESS\learning curves\IRENA summary.xlsx',
                             sheet_name="pv_cost_region")
        print("Data loaded successfully!")
        print(f"Data shape: {data.shape}")
        print("\nFirst few rows:")
        print(data.head())
        print(f"\nColumns: {list(data.columns)}")

        # Check if the data has a year column or if years are the index
        if 'Year' in data.columns or 'year' in data.columns:
            # If there's a Year column, set it as index
            year_col = 'Year' if 'Year' in data.columns else 'year'
            data = data.set_index(year_col)
        elif data.index.name in ['Year', 'year'] or all(isinstance(x, (int, float)) for x in data.index[:5]):
            # Years are already the index
            pass
        else:
            # Assume first column contains years
            data = data.set_index(data.columns[0])

        print(f"\nFinal data shape: {data.shape}")
        print(f"Index (years): {list(data.index)}")
        print(f"Columns (regions): {list(data.columns)}")

        return data

    except FileNotFoundError:
        print("ERROR: Excel file not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"ERROR reading Excel file: {str(e)}")
        return None


def exponential_interpolate(df):
    """Simple exponential interpolation between known values"""

    df_interpolated = df.copy()
    years = list(df.index)

    for region in df.columns:
        region_data = df[region].copy()

        # Get all valid (non-NaN) data points
        valid_data = region_data.dropna()
        if len(valid_data) < 2:
            continue

        valid_years = list(valid_data.index)
        valid_values = list(valid_data.values)

        # Interpolate between each pair of known values
        for i in range(len(valid_years) - 1):
            start_year = valid_years[i]
            end_year = valid_years[i + 1]
            start_value = valid_values[i]
            end_value = valid_values[i + 1]

            # Skip if consecutive years (no gap to fill)
            if end_year - start_year <= 1:
                continue

            # Calculate exponential decay rate
            num_years = end_year - start_year
            if start_value > 0 and end_value > 0:
                # r = (end_value / start_value)^(1/num_years) - 1
                growth_rate = (end_value / start_value) ** (1 / num_years) - 1
            else:
                # Fallback to linear if negative values
                growth_rate = (end_value - start_value) / (start_value * num_years)

            # Fill in missing years
            for year in years:
                if start_year < year < end_year and pd.isna(region_data[year]):
                    years_elapsed = year - start_year
                    interpolated_value = start_value * (1 + growth_rate) ** years_elapsed
                    df_interpolated.loc[year, region] = interpolated_value

                    print(f"Interpolated {region} {year}: {interpolated_value:.0f}")

    return df_interpolated


def plot_comparison(df_original, df_interpolated, regions_to_plot=None):
    """Plot comparison between original and interpolated data"""

    if regions_to_plot is None:
        # Select first 6 regions that have data
        regions_with_data = []
        for region in df_original.columns:
            if df_original[region].notna().sum() >= 3:  # At least 3 data points
                regions_with_data.append(region)
        regions_to_plot = regions_with_data[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, region in enumerate(regions_to_plot):
        if i >= len(axes):
            break

        ax = axes[i]

        # Plot original data
        original = df_original[region].dropna()
        ax.plot(original.index, original.values, 'bo-', label='Original', linewidth=2, markersize=6)

        # Plot interpolated data
        interpolated = df_interpolated[region].dropna()
        ax.plot(interpolated.index, interpolated.values, 'r--', label='Interpolated', linewidth=2, alpha=0.7)

        # Highlight interpolated points
        missing_years = df_original[region][df_original[region].isna()].index
        if len(missing_years) > 0:
            missing_values = df_interpolated[region][missing_years].dropna()
            ax.plot(missing_values.index, missing_values.values, 'rs', markersize=8, label='Filled values')

        ax.set_title(f'{region}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Installed Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis to start from 0 for better comparison
        ax.set_ylim(bottom=0)

    # Remove extra subplots
    for i in range(len(regions_to_plot), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('cost_interpolation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function"""

    print("=== Simple Exponential Cost Interpolation ===\n")

    # Parse data
    print("1. Loading data...")
    df_original = parse_data()

    if df_original is None:
        print("Failed to load data. Exiting...")
        return None, None

    print(f"Missing values before: {df_original.isna().sum().sum()}")

    # Simple exponential interpolation
    print("\n2. Performing exponential interpolation...")
    df_interpolated = exponential_interpolate(df_original)

    remaining_missing = df_interpolated.isna().sum().sum()
    print(f"\nMissing values after: {remaining_missing}")

    # Show example
    print("\n3. Example results:")
    sample_region = df_original.columns[0]
    print(f"\n{sample_region}:")
    for year in df_interpolated.index:
        if year >= 2020:  # Show recent years
            orig_val = df_original.loc[year, sample_region] if not pd.isna(
                df_original.loc[year, sample_region]) else None
            interp_val = df_interpolated.loc[year, sample_region]

            if orig_val is not None:
                print(f"  {year}: {interp_val:.0f} (Original)")
            else:
                print(f"  {year}: {interp_val:.0f} (Interpolated)")

    # Save results
    print("\n5. Saving results...")
    df_interpolated.round(0).to_csv(r'C:\Users\barna\OneDrive\Documents\Solar_BESS\learning curves\interpolated_costs.csv')
    print("Saved interpolated data to: interpolated_costs.csv")

    # Plot comparison
    print("\n5. Creating plots...")
    plot_comparison(df_original, df_interpolated)

    print("\n=== Done! ===")

    return df_original, df_interpolated


if __name__ == "__main__":
    df_original, df_interpolated = main()