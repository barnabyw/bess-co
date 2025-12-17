from pathlib import Path
import pandas as pd
import logging
import os
from functools import lru_cache

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logger = logging.getLogger("reader")

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
CWD = Path(__file__).resolve().parent
_DATA_DIR = os.path.join(CWD.parent.parent, "mappings")

# ---------------------------------------------------------
# Load mappings
# ---------------------------------------------------------
REGION_MAP = pd.read_csv(os.path.join(_DATA_DIR, "region_map.csv"))
REGION_MAP["country"] = REGION_MAP["country"].astype(str).str.strip().str.casefold()
REGION_MAP = REGION_MAP.set_index("country")

VARIABLE_REGION_MAP = pd.read_csv(
    os.path.join(_DATA_DIR, "variable_region_map.csv")
)

# Normalise lookup table
for col in ["variable", "tech", "key_type", "key", "region"]:
    VARIABLE_REGION_MAP[col] = (
        VARIABLE_REGION_MAP[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _norm_str(x):
    x = (x or "").strip()
    return x.casefold() if x else None


def _match_series(df, region, variable, tech, year, value_col):
    """
    Match rows for region / variable / tech / year.

    Assumes df['year'] contains only:
      - int years
      - or literal 'all'
    """
    if not region:
        return pd.Series(dtype="float64")

    mask = (
        df["region"].astype(str).str.casefold().eq(region)
        & df["variable"].astype(str).str.casefold().eq(variable)
    )

    if tech and "tech" in df.columns:
        mask &= df["tech"].astype(str).str.casefold().eq(tech)

    if "year" in df.columns:
        if year == "all":
            mask &= df["year"].astype(str).str.casefold().eq("all")
        else:
            mask &= df["year"].eq(year)

    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found")

    return df.loc[mask, value_col]


# ---------------------------------------------------------
# Resolve lookup regions
# ---------------------------------------------------------
def resolve_regions(
    *,
    country,
    variable,
    tech,
    region_map,
    variable_region_map,
):
    regions = []

    # ----------------------------------
    # 1) Country-specific capex region
    # ----------------------------------
    if (
        country in region_map.index
        and variable in {"capex", "capex_e", "capex_p"}
        and tech is not None
    ):
        col = f"{tech}_capex_region"
        if col in region_map.columns:
            reg = _norm_str(region_map.at[country, col])
            if reg:
                regions.append(reg)

    # ----------------------------------
    # 2) Variable-region-map rules
    # ----------------------------------
    attrs = {
        "country": country,
        "continent": (
            _norm_str(region_map.at[country, "continent"])
            if country in region_map.index else None
        ),
        "default": "world",
    }

    rules = variable_region_map[
        variable_region_map["variable"] == variable
    ]

    if tech:
        rules = rules[(rules["tech"] == tech) | (rules["tech"] == "")]
    else:
        rules = rules[rules["tech"] == ""]

    for _, row in rules.iterrows():
        key_type = row["key_type"]
        key = row["key"]
        val = attrs.get(key_type)
        if val and val == key:
            regions.append(row["region"])

    # ----------------------------------
    # 3) World fallback
    # ----------------------------------
    regions.append("world")

    return list(dict.fromkeys(regions))


# =========================================================
# 🔥 CACHED CORE LOOKUP (PURE FUNCTION)
# =========================================================
@lru_cache(maxsize=200_000)
def _get_val_cached(
    country,
    year,
    variable,
    tech,
    scenario,
    value_col,
):
    """
    Cached, pure lookup.
    Returns (value, region_used)
    """

    df = _CAPEX_OPEX_DF

    # Scenario filtering
    if "scenario" in df.columns:
        if scenario is None:
            df = df[
                df["scenario"].isna()
                | (df["scenario"].astype(str).str.strip() == "")
            ]
        else:
            df = df[
                df["scenario"].astype(str).str.casefold()
                == scenario
            ]

        if df.empty:
            raise ValueError("No rows after scenario filtering")

    regions = resolve_regions(
        country=country,
        variable=variable,
        tech=tech,
        region_map=REGION_MAP,
        variable_region_map=VARIABLE_REGION_MAP,
    )

    for idx, region in enumerate(regions):

        vals = _match_series(df, region, variable, tech, year, value_col)

        if vals.empty:
            continue

        if len(vals) > 1:
            val = float(vals.astype(float).mean())
        else:
            val = float(vals.iloc[0])

        return val, (region if idx > 0 else None)

    raise ValueError(
        f"No match found for country={country}, variable={variable}, "
        f"tech={tech}, year={year}"
    )


# =========================================================
# PUBLIC API (UNCHANGED)
# =========================================================
def get_val(
    df,
    country,
    year,
    variable,
    tech=None,
    value_col="value",
    scenario=None,
    used_fallbacks=None,
):

    global _CAPEX_OPEX_DF
    _CAPEX_OPEX_DF = df  # set once per process (safe)

    country = _norm_str(country) or ""
    variable = _norm_str(variable) or ""
    tech = _norm_str(tech)
    scenario = _norm_str(scenario)

    val, fallback_region = _get_val_cached(
        country,
        year,
        variable,
        tech,
        scenario,
        value_col,
    )

    if used_fallbacks is not None and fallback_region is not None:
        used_fallbacks[(country, variable, tech, year)] = fallback_region
        logger.info(
            f"Fallback used for {country}/{variable}/{tech}/{year}: {fallback_region}"
        )

    return val


# ---------------------------------------------------------
# Self-test
# ---------------------------------------------------------
if __name__ == "__main__":

    INPUT_PATH = os.path.join(CWD.parent.parent, "inputs")
    df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))

    print(get_val(df, "Chile", 2025, "capex_e", "BESS"))
    print(get_val(df, "Chile", 2025, "capex_e", "BESS"))  # cached
