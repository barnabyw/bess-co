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


def calculate_region_learning_rate(region_data, global_rate):
    """Calculate region-specific learning rate with fallback to global rate"""
    valid_data = region_data.dropna()
    if len(valid_data) < 3:
        return global_rate

    # Calculate year-over-year rates
    rates = []
    for i in range(1, len(valid_data)):
        if valid_data.iloc[i - 1] > 0:  # Avoid division by zero
            rate = (valid_data.iloc[i] - valid_data.iloc[i - 1]) / valid_data.iloc[i - 1]
            rates.append(rate)

    if len(rates) == 0:
        return global_rate

    # Use median to avoid outlier influence
    region_rate = np.median(rates)

    # Constrain to reasonable bounds (between -20% and +5% per year)
    region_rate = max(-0.2, min(0.05, region_rate))

    # Blend with global rate for stability
    return 0.6 * region_rate + 0.4 * global_rate


def smooth_interpolation(values, years, smoothing_factor=0.3):
    """Apply smoothing to reduce sudden jumps"""
    smoothed = values.copy()

    for i in range(1, len(values) - 1):
        # Calculate expected value based on neighbors
        prev_val = values[i - 1]
        next_val = values[i + 1]
        current_val = values[i]

        # Expected value based on linear trend between neighbors
        expected = (prev_val + next_val) / 2

        # Apply smoothing
        smoothed[i] = current_val * (1 - smoothing_factor) + expected * smoothing_factor

    return smoothed


def interpolate_missing_values(df, learning_rate):
    """Interpolate missing values with improved smoothing and trend consistency"""

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

        # Calculate region-specific learning rate
        region_learning_rate = calculate_region_learning_rate(region_data, learning_rate)

        # Step 1: Fill gaps using spline interpolation for smoothness
        valid_years = []
        valid_values = []

        for year in years:
            if year >= first_year and year <= last_year:
                if not pd.isna(region_data[year]):
                    valid_years.append(year)
                    valid_values.append(region_data[year])

        if len(valid_years) >= 3:
            # Use spline interpolation for smooth curves
            from scipy.interpolate import UnivariateSpline

            # Create spline with appropriate smoothing
            spline = UnivariateSpline(valid_years, valid_values, s=len(valid_years) * 0.1,
                                      k=min(3, len(valid_years) - 1))

            # Fill missing values
            for year in years:
                if year >= first_year and year <= last_year and pd.isna(region_data[year]):
                    interpolated_value = float(spline(year))

                    # Ensure interpolated values follow general declining trend
                    if len(valid_years) >= 2:
                        # Find surrounding years
                        prev_years = [y for y in valid_years if y < year]
                        next_years = [y for y in valid_years if y > year]

                        if prev_years and next_years:
                            prev_year = max(prev_years)
                            next_year = min(next_years)
                            prev_value = region_data[prev_year]
                            next_value = region_data[next_year]

                            # Ensure monotonic decline if that's the trend
                            if prev_value > next_value:  # Declining trend
                                max_allowed = prev_value * (1 + region_learning_rate * (year - prev_year))
                                min_allowed = next_value * (1 + region_learning_rate * (next_year - year))
                                interpolated_value = max(min_allowed, min(max_allowed, interpolated_value))

                    df_interpolated.loc[year, region] = interpolated_value

        else:
            # Fallback to linear interpolation for regions with few data points
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
                        prev_value = region_data[prev_year]
                        next_value = region_data[next_year]

                        # Simple linear interpolation
                        total_years = next_year - prev_year
                        years_from_prev = year - prev_year
                        interpolated_value = prev_value + (next_value - prev_value) * (years_from_prev / total_years)

                        df_interpolated.loc[year, region] = interpolated_value

        # Step 2: Apply trend-aware extrapolation for missing end years
        if last_year < max(years):
            last_value = df_interpolated.loc[last_year, region]

            # Look at recent trend for extrapolation
            recent_years = [y for y in valid_indices if y >= last_year - 3]
            if len(recent_years) >= 2:
                recent_values = [df_interpolated.loc[y, region] for y in recent_years]
                # Calculate trend from recent data
                recent_rate = (recent_values[-1] - recent_values[0]) / (recent_years[-1] - recent_years[0]) / \
                              recent_values[0]
                # Use more conservative extrapolation rate
                extrapolation_rate = max(-0.15, min(0.05, recent_rate))
            else:
                extrapolation_rate = region_learning_rate

            for year in range(last_year + 1, max(years) + 1):
                if year in years:
                    years_ahead = year - last_year
                    extrapolated_value = last_value * (1 + extrapolation_rate) ** years_ahead
                    # Ensure values don't go negative or increase unreasonably
                    extrapolated_value = max(0, extrapolated_value)
                    df_interpolated.loc[year, region] = extrapolated_value

        # Step 3: Apply final smoothing to reduce jumps
        region_series = df_interpolated[region].dropna()
        if len(region_series) > 2:
            smoothed_values = smooth_interpolation(region_series.values, region_series.index.values)
            for i, year in enumerate(region_series.index):
                df_interpolated.loc[year, region] = smoothed_values[i]

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

    if df_original is None:
        print("Failed to load data. Exiting...")
        return None, None

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
    print("\n5. Sample interpolated values:")
    # Use the first available region for sample display
    sample_region = df_original.columns[0]
    print(f"{sample_region} sample years:")
    for year in [2020, 2021, 2022, 2023, 2024]:
        if year in df_interpolated.index and sample_region in df_interpolated.columns:
            orig = df_original.loc[year, sample_region] if year in df_original.index else np.nan
            interp = df_interpolated.loc[year, sample_region]
            status = "Original" if not pd.isna(orig) else "Interpolated"
            print(f"  {year}: {interp:.0f} ({status})")

    # Save results
    print("\n5. Saving results...")
    df_interpolated.round(0).to_csv(r'C:\Users\barna\OneDrive\Documents\Solar_BESS\learning curves\interpolated_costs.csv')
    print("Saved interpolated data to: interpolated_costs.csv")

    # Create comparison plot
    print("\n7. Creating comparison plots...")
    plot_comparison(df_original, df_interpolated)

    print("\n=== Interpolation Complete ===")

    return df_original, df_interpolated


if __name__ == "__main__":
    df_original, df_interpolated = main()