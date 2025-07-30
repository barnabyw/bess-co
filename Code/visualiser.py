import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from matplotlib.animation import FuncAnimation
import contextily as ctx

# Generate sample data
np.random.seed(0)
num_points = 50
timestamps = pd.date_range("2023-01-01", periods=10, freq="D")
data = []

for t in timestamps:
    lats = np.random.uniform(51.4, 51.6, num_points)
    lons = np.random.uniform(-0.2, 0.1, num_points)
    values = np.random.randint(0, 100, num_points)
    for lat, lon, val in zip(lats, lons, values):
        data.append({"timestamp": t, "latitude": lat, "longitude": lon, "value": val})

df = pd.DataFrame(data)

# Create square grid cells
cell_size = 0.01
df["geometry"] = df.apply(lambda row: box(
    row["longitude"] - cell_size/2,
    row["latitude"] - cell_size/2,
    row["longitude"] + cell_size/2,
    row["latitude"] + cell_size/2
), axis=1)

gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)  # Web Mercator for basemap

# Ensure the GeoDataFrame has the required columns
if 'timestamp' not in gdf.columns or 'value' not in gdf.columns:
    raise ValueError("GeoDataFrame must contain 'timestamp' and 'value' columns.")

# Convert to Web Mercator for compatibility with contextily
gdf = gdf.to_crs(epsg=3857)

# Determine color scale limits
vmin, vmax = gdf["value"].min(), gdf["value"].max()

# Select the first timestamp
first_timestamp = gdf["timestamp"].min()
subset = gdf[gdf["timestamp"] == first_timestamp]

# Plot the choropleth map
fig, ax = plt.subplots(figsize=(10, 10))
subset.plot(column="value", cmap="viridis", linewidth=0.1, ax=ax, edgecolor="black", vmin=vmin, vmax=vmax)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title(f"Choropleth Map for {first_timestamp.date()}")
ax.axis("off")

plt.tight_layout()
plt.show()
