import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def parse_data():
    """Parse the raw data and create a structured DataFrame"""
    data = pd.read_excel(
        r'C:\Users\barna\OneDrive\Documents\Solar_BESS\learning curves\IRENA summary.xlsx',
        sheet_name="pv_cost_region"
    )

    # Ensure 'Year' is the index
    if "Year" in data.columns:
        data = data.set_index("Year")

    return data


def calculate_global_learning_rate(df):
    """Calculate global learning rate from available data"""

    # Calculate global average for each year (excluding NaN values)
    global_avg = df.mean(axis=1, skipna=True)

    # Calculate year-over-year learning rates
    learning_rates = []
    years = list(global_avg.index)

    for i in range(1, len(years)):
        if not pd.isna(global_avg.iloc[i - 1]) and not pd.isna(global_avg.iloc[i]):
            rate = (global_avg.iloc[i] - global_avg.iloc[i - 1]) / global_avg.iloc[i - 1]
            learning_rates.append(rate)

    # Calculate average learning rate
    avg_learning_rate = np.mean(learning_rates)
    print(f"Global average learning rate: {avg_learning_rate:.4f} ({avg_learning_rate * 100:.2f}% per year)")

    return avg_learning_rate, global_avg


def interpolate_missing_values(df, learning_rate):
    """Interpolate missing values using global learning rate and region-specific patterns"""

    df_interpolated = df.copy()
    years = list(df.index)

    for region in df.columns:
        region_data = df[region].copy()

        # Find first and last valid values
        valid_indices = region_data.dropna().index
        if len(valid_indices) < 2:
            continue  # Skip if less than 2 data points

        first_year = valid_indices[0]
        last_year = valid_indices[-1]

        # Interpolate missing values between first and last valid years
        for year in years:
            if year >= first_year and year <= last_year and pd.isna(region_data[year]):

                # Find nearest valid values
                prev_year = None
                next_year = None

                for y in reversed(years[:years.index(year)]):
                    if not pd.isna(region_data[y]):
                        prev_year = y
                        break

                for y in years[years.index(year) + 1:]:
                    if not pd.isna(region_data[y]):
                        next_year = y
                        break

                if prev_year is not None and next_year is not None:
                    # Linear interpolation with learning rate adjustment
                    prev_value = region_data[prev_year]
                    next_value = region_data[next_year]

                    # Calculate time-weighted interpolation
                    total_years = next_year - prev_year
                    years_from_prev = year - prev_year

                    # Apply learning rate trend
                    expected_value = prev_value * (1 + learning_rate) ** years_from_prev

                    # Blend with simple linear interpolation
                    linear_value = prev_value + (next_value - prev_value) * (years_from_prev / total_years)

                    # Weighted combination (70% learning rate, 30% linear)
                    interpolated_value = 0.7 * expected_value + 0.3 * linear_value

                    df_interpolated.loc[year, region] = interpolated_value

        # Extrapolate forward from last known value if needed
        if last_year < 2024:
            last_value = region_data[last_year]
            for year in range(last_year + 1, 2025):
                if year in years:
                    years_ahead = year - last_year
                    extrapolated_value = last_value * (1 + learning_rate) ** years_ahead
                    df_interpolated.loc[year, region] = extrapolated_value

    return df_interpolated


def plot_comparison(df_original, df_interpolated, regions_to_plot=None):
    """Plot comparison between original and interpolated data"""

    if regions_to_plot is None:
        # Select a few interesting regions for plotting
        regions_to_plot = ['Europe', 'Africa', 'China', 'United States', 'Germany', 'India', 'Brazil']

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
            missing_values = df_interpolated[region][missing_years]
            ax.plot(missing_years, missing_values, 'rs', markersize=8, label='Filled values')

        ax.set_title(f'{region}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Installed Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplots
    for i in range(len(regions_to_plot), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('cost_interpolation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function"""

    print("=== Cost Data Interpolation Script ===\n")

    # Parse data
    print("1. Parsing input data...")
    df_original = parse_data()
    print(f"Data shape: {df_original.shape}")
    print(f"Missing values: {df_original.isna().sum().sum()}")

    # Calculate global learning rate
    print("\n2. Calculating global learning rate...")
    learning_rate, global_avg = calculate_global_learning_rate(df_original)

    # Interpolate missing values
    print("\n3. Interpolating missing values...")
    df_interpolated = interpolate_missing_values(df_original, learning_rate)
    filled_values = df_interpolated.isna().sum().sum()
    print(f"Remaining missing values: {filled_values}")

    # Display sample results
    print("\n4. Sample interpolated values:")
    print("China (2016-2019):")
    for year in list(range(2015,2025)):
        if year in df_interpolated.index:
            orig = df_original.loc[year, 'China']
            interp = df_interpolated.loc[year, 'China']
            status = "Original" if not pd.isna(orig) else "Interpolated"
            print(f"  {year}: {interp:.0f} ({status})")

    # Save results
    print("\n5. Saving results...")
    df_interpolated.round(0).to_csv(r'C:\Users\barna\OneDrive\Documents\Solar_BESS\learning curves\interpolated_costs.csv')
    print("Saved interpolated data to: interpolated_costs.csv")

    # Create comparison plot
    print("\n6. Creating comparison plots...")
    plot_comparison(df_original, df_interpolated)

    print("\n=== Interpolation Complete ===")

    return df_original, df_interpolated


if __name__ == "__main__":
    df_original, df_interpolated = main()