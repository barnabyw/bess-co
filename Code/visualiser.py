from assumptions import output_path, input_path

import os
import pandas as pd
import plotly.express as px
import numpy as np

df = pd.read_csv(os.path.join(output_path, "multi_yearly_results.csv"))

## Create log-transformed LCOE
df["log_LCOE"] = np.log10(df["LCOE"])

# Create choropleth
fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="log_LCOE",
    animation_frame="Year",
    hover_name="Country",
    custom_data=["LCOE", "Year"],  # Include both LCOE and Year in custom data
    color_continuous_scale="Viridis_r",
    range_color=(df["log_LCOE"].min(), df["log_LCOE"].max()),
    title="LCOE by Country (Log Scale, Animated Over Time)"
)

# Define readable tick labels for the color bar
tick_vals = np.arange(np.floor(df["log_LCOE"].min()), np.ceil(df["log_LCOE"].max()) + 1)
tick_text = [f"{10**val:.0f}" for val in tick_vals]

# Update layout with custom colorbar and hover template
fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="white",      # Coastlines
        coastlinewidth=1,
        showcountries=True,          # Enable country borders
        countrycolor="white",        # Country borders
        countrywidth=1,              # Country border width
        showlakes=True,
        lakecolor="lightblue"
    ),
    coloraxis_colorbar=dict(
        title="LCOE ($/MWh)",
        tickvals=tick_vals,
        ticktext=tick_text
    ),
    margin=dict(l=0, r=0, t=40, b=0),
)

# Customize hover template to show original LCOE and Year
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>" +
                  "LCOE: %{customdata[0]:.2f} $/MWh<br>" +
                  "Year: %{customdata[1]}<br>" +
                  "<extra></extra>",  # Removes the trace box
    marker_line_color="white",        # Country borders
    marker_line_width=0.5               # Border thickness
)

fig.show()