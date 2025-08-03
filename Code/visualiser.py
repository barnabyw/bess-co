from assumptions import output_path, input_path

import os
import pandas as pd
import plotly.express as px
import numpy as np

df = pd.read_csv(os.path.join(output_path, "multi_yearly_results.csv"))

# Create a new column for log-transformed LCOE
df["log_LCOE"] = np.log10(df["LCOE"])

# Create animated choropleth using the log-transformed data
fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="log_LCOE",
    animation_frame="Year",
    hover_name="Country",
    color_continuous_scale="Viridis",
    range_color=(df["log_LCOE"].min(), df["log_LCOE"].max()),
    title="LCOE by Country (Log Scale, Animated Over Time)"
)

# Update the colorbar to show the original LCOE scale (anti-log ticks)
tick_vals = np.arange(np.floor(df["log_LCOE"].min()), np.ceil(df["log_LCOE"].max()) + 1)
tick_text = [f"{10**val:.0f}" for val in tick_vals]

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True),
    coloraxis_colorbar=dict(
        title="LCOE ($/MWh)",
        tickvals=tick_vals,
        ticktext=tick_text
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.show()
