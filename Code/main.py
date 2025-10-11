from profile import generate_hourly_solar_profile
from optimiser import optimise_bess
from assumptions import *

# Import data
data = pd.read_csv(os.path.join(input_path, "all_country_coordinates_2.csv"))

results = []

for _, row in data.iterrows():
    country = row["Country"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    print(f"Processing {country}...")
    yearly_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)

    # use mid year to determine optimum ratio
    year = 2024
    cost, solar_cap, bess_energy, lev_cost, interval_results = optimise_bess(yearly_profile, capex_learning_df, year)

    # Store or print results as needed
    results.append({
        "Country": country,
        "Latitude": lat,
        "Longitude": lon,
        "LCOE": lev_cost,
        "Cost": cost,
        "Year": year,
        "Solar_Capacity": solar_cap,
        "BESS_Energy": bess_energy
    })

# Convert to DataFrame if needed
df_results = pd.DataFrame(results)

print(df_results)
df_results.to_csv(os.path.join(output_path, "results80.csv"), index=False)