from profile import generate_hourly_solar_profile
from optimiser import optimise_bess
from assumptions import *
from lcoe import lcoe
import matplotlib.pyplot as plt

import numpy as np
data = pd.read_csv(os.path.join(input_path, "all_country_coordinates_2.csv"))

results = []
countries = ['United Kingdom', 'Kenya']
avail = [0.5,0.6,0.7,0.8,0.9]

for country in countries:
    data_filtered = data[data['Country'] == country]
    lat = data_filtered['Latitude'].values[0]
    lon = data_filtered['Longitude'].values[0]
    print(f"For {country}, Latitude: {lat}, Longitude: {lon}")
    solar_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)
    for a in avail:
        cost, solar_capacity, bess_energy, levcost = optimise_bess(solar_profile, capex_learning_df, 2024, availability=a)
        results.append({
            "Country": country,
            "Availability": a,
            "LCOE": levcost
        })

# Convert results to DataFrame
df_results = pd.DataFrame(results)
file_path = r'C:\\Users\\barna\OneDrive\Documents\Solar_BESS results\lcoe_availability.csv'
df_results.to_csv(file_path, index=False)

# Create quad subplot
fig, axes = plt.subplots(2, 1, figsize=(15, 12))
fig.suptitle('LCOE vs Availability by Country', fontsize=16, fontweight='bold')

# Flatten axes for easier indexing
axes_flat = axes.flatten()

# Plot each country
for i, country in enumerate(countries):
    country_data = df_results[df_results['Country'] == country]

    axes_flat[i].plot(country_data['Availability'], country_data['LCOE'],
                      marker='o', linewidth=2, markersize=8,
                      label=country, color=f'C{i}')

    axes_flat[i].set_title(f'{country}', fontsize=14, fontweight='bold')
    axes_flat[i].set_xlabel('Availability', fontsize=12)
    axes_flat[i].set_ylabel('LCOE ($/MWh)', fontsize=12)
    axes_flat[i].grid(True, alpha=0.3)

    # Add some styling
    axes_flat[i].tick_params(axis='both', which='major', labelsize=10)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.93)  # Make room for main title

# Show the plot
plt.show()

"""
if __name__ == "__main__":

    latitude = 19.4326
    longitude = 99.1332

    # Create a list to store the results
    yearly_lcoe_results = []

    # Call the placeholder functions with the current year
    solar_profile = generate_hourly_solar_profile(latitude, longitude, solar_year=2023)
    capex_df = capex_learning_df

    cost, solar_capacity, bess_energy, levcost = optimise_bess(solar_profile, capex_learning_df, 2010)

    # Loop through the years from 2010 to 2024
    for year in range(2010, 2025):
        print(f"--- Processing year: {year} ---")

        solar_cost_per_mw = capex_df.loc[capex_df["year"] == year, "solar_cost_per_mw"].values[0]
        bess_energy_cost_per_mwh = capex_df.loc[capex_df["year"] == year, "bess_energy_cost_per_mwh"].values[0]

        cost = solar_capacity * solar_cost_per_mw + bess_energy * bess_energy_cost_per_mwh
        levcost = 1000 * lcoe.lcoe(load * 8760 * target, cost, 0, 0.08, 20)
        print(f"LCOE for {year}: {levcost:.4f}")

        # Append the results to the list
        yearly_lcoe_results.append({'Year': year, 'LCOE': levcost})

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(yearly_lcoe_results)

    # Save the results to a CSV file
    file_path = r'C:\\Users\\barna\OneDrive\Documents\Solar_BESS results\yearly_lcoe_results.csv'
    results_df.to_csv(file_path, index=False)

    print("\n--------------------")
    print("Final Results:")
    print(results_df)
    print(f"\nResults saved to '{file_path}'")
"""