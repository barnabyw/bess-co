import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location

def generate_hourly_solar_profile(latitude, longitude, year=2024):
    # Define location
    site = Location(latitude, longitude)

    # Generate hourly times for the year
    times = pd.date_range(start=f'{year}-01-01', end=f'{year}-06-30 23:00:00', freq='h', tz=site.tz)

    # Get solar position and clear-sky irradiance
    solar_position = site.get_solarposition(times)
    clearsky = site.get_clearsky(times)

    # Use GHI (Global Horizontal Irradiance) as a proxy for solar availability
    ghi = clearsky['ghi']

    # Normalize to max value to get availability factor (0 to 1)
    normalized_output = ghi / ghi.max()
    return normalized_output.values  # returns a NumPy array of 8760 values
