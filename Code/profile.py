import pandas as pd
from pvlib.location import Location
from pvlib.iotools import get_pvgis_hourly
from requests.exceptions import HTTPError


def generate_hourly_historical_solar_profile(latitude, longitude, solar_year):
    """
    Generates an hourly, historical solar availability profile using
    PVGIS databases, automatically falling back to ERA5 if SARAH/NSRDB fails.

    Args:
        latitude (float): Latitude of the site (e.g., 51.5).
        longitude (float): Longitude of the site (e.g., 0.1).
        start_year (int): The first year of the historical period to analyze.
        end_year (int): The last year of the historical period to analyze.

    Returns:
        pandas.Series: A time-series of hourly solar availability factors (0 to 1).

    Raises:
        HTTPError: If even the global ERA5 database fails (e.g., polar regions).
    """

    # 1. Define the search order for databases
    # SARAH3/SARAH2/NSRDB have higher spatial resolution but limited coverage (e.g., not polar).
    # ERA5 is a re-analysis product with global coverage (except poles).
    databases = ["PVGIS-SARAH3", "PVGIS-ERA5", "PVGIS-NSRDB"]

    data = None
    meta = None

    # Common parameters for the PVGIS API call
    pvgis_params = {
        'latitude': latitude,
        'longitude': longitude,
        'start': solar_year,
        'end': solar_year,
        'outputformat': 'json',
        'components': False,
        'peakpower': 1,
        'pvtechchoice': 'crystSi',
        'optimal_surface_tilt': True,
        'optimalangles': True,
        'pvcalculation': True
    }

    print(f"Attempting to fetch data for ({latitude:.2f}, {longitude:.2f})...")

    for db_name in databases:
        try:
            print(f"  Attempting database: {db_name}...")
            # Attempt API call with the current database
            data, meta = get_pvgis_hourly(raddatabase=db_name, **pvgis_params)

            # If successful, break the loop
            print(f"  ✅ Successfully retrieved data using {db_name}.")
            break

        except HTTPError as e:
            # Check if the error is the expected coverage error
            if "spatial coverage" in str(e):
                print(f"  ❌ {db_name} failed due to spatial coverage. Trying next...")
                continue  # Move to the next database in the list
            else:
                # If it's a different HTTP error (e.g., rate limit, server error), raise it
                raise e
        except Exception as e:
            # Catch other potential errors (e.g., network timeout)
            print(f"  An unexpected error occurred with {db_name}: {e}. Trying next...")
            continue

    # 2. Check if data was successfully retrieved
    if data is None:
        raise ConnectionError(
            "All PVGIS databases failed for this location. Check coordinates or PVGIS availability."
        )

    output = data['P']
    af = output/1000

    return af.values


# Example of how to use it:
# profile = generate_hourly_historical_solar_profile(latitude=70.0, longitude=25.0, start_year=2020, end_year=2020)
# This location is near the Arctic Circle and would definitely fail SARAH3, forcing a switch to ERA5.

def generate_hourly_solar_profile(latitude, longitude, solar_year=2024):
    # Define location
    site = Location(latitude, longitude)

    # Generate hourly times for the year
    times = pd.date_range(start=f'{solar_year}-01-01', end=f'{solar_year}-12-31 23:00:00', freq='h', tz=site.tz)

    # Get solar position and clear-sky irradiance
    solar_position = site.get_solarposition(times)
    clearsky = site.get_clearsky(times)

    # Use GHI (Global Horizontal Irradiance) as a proxy for solar availability
    ghi = clearsky['ghi']

    # Normalize to max value to get availability factor (0 to 1)
    normalized_output = ghi / ghi.max()
    return normalized_output.values  # returns a NumPy array of 8760 values


def parse_renewables_ninja(filepath):
    """
    Parse Renewables.ninja CSV file and return normalized wind power data.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        np.array: Array of normalized wind power values (0 to 1)
    """
    # Read file and skip first 3 lines
    df = pd.read_csv(filepath, skiprows=3)

    # Convert electricity column to numpy array and normalize
    electricity_values = df['electricity'].values
    normalized_values = electricity_values / electricity_values.max()

    return normalized_values

if __name__ == "__main__":
    latitude = 19.4326
    longitude = 99.1332

    df = generate_hourly_historical_solar_profile(latitude, longitude, solar_year=2023)
    #df = parse_renewables_ninja(r"C:\Users\barna\Downloads\ninja_wind_54.7867_-1.9809_corrected.csv")
    print(df)