import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.animation import FuncAnimation

# --- Example time series ---
time = np.linspace(0, 10, 200)
y1 = np.sin(time)
y2 = 0.5 * np.cos(time)
diff = y1 - y2

# --- Load shapefile (built-in from GeoPandas) ---
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
country = world[world.name == "France"].copy()

# --- Set up subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Left: line plot
ax1.plot(time, y1, label="Series 1")
ax1.plot(time, y2, label="Series 2")
vline = ax1.axvline(time[0], color="k", linestyle="--")  # moving vertical line
ax1.legend()
ax1.set_title("Line plot")

# Right: map
country.plot(ax=ax2, color="lightgrey", edgecolor="black")  # base map
map_artist = None  # placeholder
ax2.axis("off")
ax2.set_title("Difference map")


# --- Update function for animation ---
def update(frame):
    global map_artist
    # Update vertical line on plot
    vline.set_xdata([time[frame], time[frame]])

    # Remove previous colored patch if exists
    if map_artist is not None:
        map_artist.remove()

    # Update map with new diff value
    country["diff"] = diff[frame]
    map_artist = country.plot(
        column="diff", cmap="coolwarm", vmin=diff.min(), vmax=diff.max(),
        ax=ax2, legend=False
    )

    return vline, map_artist


# --- Animate ---
ani = FuncAnimation(fig, update, frames=len(time), interval=100, blit=False)

plt.show()
