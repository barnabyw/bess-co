import pandas as pd
import plotly.express as px

# Sample DataFrame structure
# df = pd.read_csv('your_data.csv')
# Ensure country names match ISO standard names (like those used by Plotly)

fig = px.choropleth(
    df,
    locations="country",           # Name of column with country names
    locationmode="country names",  # OR use "ISO-3" and provide ISO codes
    color="value",                 # Column to be color-coded
    animation_frame="year",        # Time dimension
    color_continuous_scale="Viridis",  # Or "Plasma", "Cividis", etc.
    range_color=(df["value"].min(), df["value"].max()),
    title="Animated Country Map Over Years",
    labels={'value': 'Your Metric'}
)

fig.update_layout(geo=dict(showframe=False, showcoastlines=False))
fig.show()
