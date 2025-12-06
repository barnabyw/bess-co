# bess_capex_cache.py
import pandas as pd

_CAPEX_CACHE = {}

def get_bess_capex_series(capex_opex_df: pd.DataFrame, scenario: str | None):
    """
    Returns a cached pandas Series indexed by year, containing capex_e values
    for the requested scenario.

    Rules:
        - scenario=None → return rows where scenario is empty ("", NaN)
        - scenario="Low" → return Low scenario rows
    """

    scenario_key = scenario or ""  # default scenario

    if scenario_key in _CAPEX_CACHE:
        return _CAPEX_CACHE[scenario_key]

    # --- filter ---
    df = capex_opex_df[
        (capex_opex_df["tech"] == "BESS") &
        (capex_opex_df["variable"] == "capex_e")
    ].copy()

    if scenario_key == "":
        df = df[df["scenario"].isna() | (df["scenario"] == "")]
    else:
        df = df[df["scenario"] == scenario_key]

    df = df[["year", "value"]].dropna()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year")

    series = df.set_index("year")["value"]

    # cache
    _CAPEX_CACHE[scenario_key] = series
    return series
