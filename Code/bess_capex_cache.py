# bess_capex_cache.py
import pandas as pd
import numpy as np
from data_prep.reader2 import get_val

_CAPEX_CACHE = {}

def get_bess_capex_series(
    capex_opex_df: pd.DataFrame,
    country: str,
    scenario: str | None = None,
):
    """
    Return a clean {year → capex_e} series for BESS energy,
    resolved per-country using get_val logic.
    Cached by (country, scenario).
    """

    cache_key = (country.casefold(), scenario)

    # ----------------------------------
    # CACHE HIT
    # ----------------------------------
    if cache_key in _CAPEX_CACHE:
        return _CAPEX_CACHE[cache_key]

    # ----------------------------------
    # CACHE MISS → build series
    # ----------------------------------
    years = capex_opex_df["year"].dropna().unique()
    years = [y for y in years if isinstance(y, (int, np.integer))]

    values = {}

    for y in years:
        try:
            values[y] = get_val(
                capex_opex_df,
                country=country,
                year=y,
                variable="capex_e",
                tech="BESS",
                scenario=scenario,
            )
        except Exception:
            continue

    series = pd.Series(values).sort_index()

    # ----------------------------------
    # STORE IN CACHE
    # ----------------------------------
    _CAPEX_CACHE[cache_key] = series

    return series
