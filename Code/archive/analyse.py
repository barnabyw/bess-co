from analyser import SolarBESSAnalyzer, OptimizationParams
from assumptions import *
import os
import pandas as pd

country_coords_df = pd.read_csv(os.path.join(input_path, "all_country_coordinates_2.csv"))

# Initialize the analyzer (target=0.8 default from assumptions)
analyzer = SolarBESSAnalyzer(capex_learning_df, country_coords_df)

# Example 1: Single location time series
lat, lon = 19.4326, 99.1332
time_series = analyzer.analyze_single_location(lat, lon, range(2010, 2025))
analyzer.save_results(time_series, 'single_location_time_series.csv')

# Example 2: Multi-year analysis with fixed capacities
results_2025 = analyzer.analyze_countries(year=2024)
multi_year_results = analyzer.analyze_multi_year_fixed_capacity(results_2025)
analyzer.save_results(multi_year_results, 'multi_year_lcoe.csv')

# Example 3: Availability sensitivity
countries = ['United Kingdom', 'Kenya']
availabilities = [0.5, 0.6, 0.7, 0.8, 0.9]  # 0–1 scale (50%–90%)
sensitivity_results = analyzer.analyze_availability(countries, availabilities)
analyzer.save_results(sensitivity_results, 'availability_sensitivity.csv')

# Example 4: All countries single year
all_countries_2025 = analyzer.analyze_countries(year=2025)
analyzer.save_results(all_countries_2025, 'all_countries_2025.csv')

# Example 5: Custom availability target
custom_params = OptimizationParams(target_availability=0.95)  # 95% availability
analyzer_high_avail = SolarBESSAnalyzer(capex_learning_df, country_coords_df, custom_params)
high_avail_results = analyzer_high_avail.analyze_countries(['United Kingdom'], year=2025)
analyzer.save_results(high_avail_results, 'uk_high_availability.csv')
