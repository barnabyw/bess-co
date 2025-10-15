from Code.yearly_data import calculate_multi_year_costs_lcoe
from profile import generate_hourly_solar_profile
from optimiser import optimise_bess
from assumptions import *
from reader import get_val

from lcoe.lcoe import lcoe

# Import data
data = pd.read_csv(os.path.join(input_path, "all_country_coordinates_2.csv"))
capex_opex = pd.read_excel(os.path.join(input_path, "capex_opex_converted_2025USD.xlsx"))
results = []

years = list(range(2010, 2025))

for _, row in data.iterrows():
    country = row["Country"]
    lat = row["Latitude"]
    lon = row["Longitude"]

    print(f"Processing {country}...")
    yearly_profile = generate_hourly_solar_profile(lat, lon, solar_year=2023)

    # use mid year to determine optimum ratio
    year = 2024

    solar_capex = get_val(capex_opex, country, year, "capex", "Solar")
    bess_capex = get_val(capex_opex, country, year, "capex", "BESS")

    cost, solar_cap, bess_energy, lev_cost, interval_results = optimise_bess(yearly_profile, solar_capex, bess_capex)

    # Store or print results as needed
    results.append({
        "Country": country,
        "Latitude": lat,
        "Longitude": lon,
        "LCOE": lev_cost,
        "Cost": cost,
        "Year": year,
        "Solar_Capacity": solar_cap,
        "BESS_Energy": bess_energy,
        "Tech": "Solar BESS"
    })

    df_results = pd.DataFrame(results)
    results = calculate_multi_year_costs_lcoe(df_results, capex_opex, years)

    coal_capex = get_val(capex_opex, country, year,"capex", "Coal")
    coal_fuel = get_val(capex_opex, country, year, "fuel", "Coal")

    gas_capex = get_val(capex_opex, country, year, "capex", "Gas")
    gas_fuel = get_val(capex_opex, country, year, "fuel", "Gas")



# Convert to DataFrame if needed
df_results = pd.DataFrame(results)

print(df_results)
df_results.to_csv(os.path.join(output_path, "results80.csv"), index=False)