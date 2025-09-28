from analyser import SolarBESSAnalyzer, OptimizationParams
from assumptions import *

country_coords_df = pd.read_csv(os.path.join(input_path, "all_country_coordinates_2.csv"))

# Initialize the analyzer (will use target=0.8 from assumptions by default)
analyzer = SolarBESSAnalyzer(capex_learning_df, country_coords_df)

# Example 2: Single location time series (replaces function 2)
lat, lon = 19.4326, 99.1332
time_series = analyzer.analyze_single_location_time_series(lat, lon, range(2010, 2025))
analyzer.save_results(time_series, 'single_location_time_series.csv')

# Example 1: Multi-year analysis with fixed capacities (replaces function 1)
results_2025 = analyzer.analyze_countries_single_year(year=2024)
multi_year_results = analyzer.analyze_multi_year_fixed_capacity(results_2025)
analyzer.save_results(multi_year_results, 'multi_year_lcoe.csv')

# Example 3: Availability sensitivity (replaces function 3)
countries = ['United Kingdom', 'Kenya']
availabilities = [0.5, 0.6, 0.7, 0.8, 0.9]  # Note: these are 0-1 scale (50%, 60%, etc.)
sensitivity_results = analyzer.analyze_availability_sensitivity(countries, availabilities)
analyzer.save_results(sensitivity_results, 'availability_sensitivity.csv')

# Example 4: All countries single year (replaces function 4)
all_countries_2025 = analyzer.analyze_countries_single_year(year=2025)
analyzer.save_results(all_countries_2025, 'all_countries_2025.csv')

# Example 5: Custom availability target
custom_params = OptimizationParams(target_availability=0.95)  # 95% availability
analyzer_high_avail = SolarBESSAnalyzer(capex_learning_df, country_coords_df, custom_params)
high_avail_results = analyzer_high_avail.analyze_countries_single_year(['United Kingdom'], year=2025)