import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location

import pandas as pd
import numpy as np
from pvlib.iotools import get_pvgis_hourly

def generate_real_hourly_solar_profile(latitude, longitude, solar_year=2023):
    """
    Downloads real GHI data for the specified location and year using PVGIS.

    Returns:
        A NumPy array of normalized hourly GHI values (availability factor between 0 and 1).
    """
    # Download hourly PVGIS data (includes GHI, DNI, DHI)
    df, meta = get_pvgis_hourly(
        latitude, longitude,
        start=solar_year, end=solar_year,
        raddatabase='PVGIS-ERA5',  # most complete for Europe, MENA
        surface_tilt=0,  # horizontal plane
        surface_azimuth=180,
        outputformat='json',
        usehorizon=True,
        components=True,
    )

    # Combine POA components
    poa_irradiance = (
        df['poa_direct'] +
        df['poa_sky_diffuse'] +
        df['poa_ground_diffuse']
    )

    # Normalize
    normalized_output = poa_irradiance / poa_irradiance.max()

    return normalized_output.values

def generate_hourly_solar_profile(latitude, longitude, solar_year=2024):
    # Define location
    site = Location(latitude, longitude)

    # Generate hourly times for the year
    times = pd.date_range(start=f'{solar_year}-01-01', end=f'{solar_year}-06-30 23:00:00', freq='h', tz=site.tz)

    # Get solar position and clear-sky irradiance
    solar_position = site.get_solarposition(times)
    clearsky = site.get_clearsky(times)

    # Use GHI (Global Horizontal Irradiance) as a proxy for solar availability
    ghi = clearsky['ghi']

    # Normalize to max value to get availability factor (0 to 1)
    normalized_output = ghi / ghi.max()
    return normalized_output.values  # returns a NumPy array of 8760 values

if __name__ == "__main__":
    latitude = 19.4326
    longitude = 99.1332

    df = generate_real_hourly_solar_profile(latitude, longitude, solar_year=2023)
