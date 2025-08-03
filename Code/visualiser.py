from assumptions import output_path, input_path

import os
import pandas as pd
import plotly.express as px

df = pd.read_csv(os.path.join(output_path, "multi_yearly_results.csv"))

# Create animated choropleth
fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",  # Match country names to map
    color="LCOE",
    animation_frame="Year",
    hover_name="Country",
    color_continuous_scale="Viridis",
    range_color=(df["LCOE"].min(), df["LCOE"].max()),
    title="LCOE by Country (Animated Over Time)"
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True),
    coloraxis_colorbar=dict(title="LCOE ($/MWh)"),
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.show()
