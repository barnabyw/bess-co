import pandas as pd
import os
import plotly.express as px
from assumptions import output_path

# Sample DataFrame structure
df = pd.read_csv(os.path.join(output_path, "multi_yearly_results.csv"))
# Ensure country names match ISO standard names (like those used by Plotly)

fig = px.choropleth(
    df,
    locations="Country",           # Name of column with country names
    locationmode="country names",  # OR use "ISO-3" and provide ISO codes
    color="LCOE",                 # Column to be color-coded
    animation_frame="Year",        # Time dimension
    color_continuous_scale="Viridis",  # Or "Plasma", "Cividis", etc.
    range_color=(100, 1000), #df["LCOE"].min()
    title="Animated Country Map Over Years",
    labels={'value': 'Your Metric'}
)

fig.update_layout(geo=dict(showframe=False, showcoastlines=False))
fig.show()
