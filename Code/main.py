from profile import generate_hourly_solar_profile
from optimiser import optimise_bess
from assumptions import *

# Import data
data = pd.read_csv(os.path.join(input_path, "country_coordinates.csv"))

results = []

for _, row in data.iterrows():
    country = row["Country"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    for year in years:
        print(f"Processing {country}...")
        yearly_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)
        cost, solar_cap, bess_power, bess_energy = optimise_bess(yearly_profile, capex_learning_df,year)

        # Store or print results as needed
        results.append({
            "Country": country,
            "Latitude": lat,
            "Longitude": lon,
            "Cost": cost,
            "Year": year,
            "Solar_Capacity": solar_cap,
            "BESS_Power": bess_power,
            "BESS_Energy": bess_energy
        })

# Convert to DataFrame if needed
df_results = pd.DataFrame(results)

print(df_results)
df_results.to_csv(os.path.join(output_path, "results.csv"), index=False)