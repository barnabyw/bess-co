from pathlib import Path
import pandas as pd
import logging
import os

# ---------------------------------------------------------
# Logging (match main_workflow.py style)
# ---------------------------------------------------------
logger = logging.getLogger("reader")

# --- Load default mappings (quietly tolerate absence) ---

# === OUTPUT PATH (MAIN INPUT) ===
CWD = Path(__file__).resolve().parent
_DATA_DIR = os.path.join(CWD.parent.parent, "mappings")

_DEFAULT_PROXY_RULES = pd.read_csv(os.path.join(_DATA_DIR, "proxy_rules.csv"))
_DEFAULT_REGION_MAP = pd.read_csv(os.path.join(_DATA_DIR, "region_map.csv"))
_DEFAULT_REGION_MAP["country"] = (
    _DEFAULT_REGION_MAP["country"].str.strip().str.casefold()
)
_DEFAULT_REGION_MAP = _DEFAULT_REGION_MAP.set_index("country")


def _norm_str(x: str | None) -> str | None:
    x = (x or "").strip()
    return x.casefold() if x else None


def _norm_year(y) -> int | None:
    if y is None:
        return None
    if isinstance(y, str) and y.strip().casefold() in {"all", "any", "*"}:
        return None
    try:
        return int(y)
    except Exception:
        return None


def _match_series(df: pd.DataFrame,
                  region: str | None,
                  variable: str,
                  tech: str | None,
                  year: int | None,
                  value_col: str) -> pd.Series:

    if not region or "region" not in df or "variable" not in df:
        return pd.Series(dtype="float64")

    reg = df["region"].astype(str).str.strip().str.casefold()
    var = df["variable"].astype(str).str.strip().str.casefold()
    mask = reg.eq(region) & var.eq(variable)

    if tech and "tech" in df.columns:
        tch = df["tech"].astype(str).str.strip().str.casefold()
        mask &= tch.eq(tech)

    if year is not None and "year" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["year"]):
            with pd.option_context("mode.chained_assignment", None):
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
        mask &= df["year"].eq(year)

    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in df.")

    return df.loc[mask, value_col]


def _proxy_region(country: str,
                  variable: str,
                  tech: str | None,
                  region_map: pd.DataFrame,
                  proxy_rules: pd.DataFrame) -> str | None:

    if proxy_rules is None or proxy_rules.empty:
        return None

    subregion = continent = None
    if country in region_map.index:
        subregion = _norm_str(region_map.at[country, "subregion"])
        continent = _norm_str(region_map.at[country, "continent"])

    pr = proxy_rules.copy()

    var_mask = pr.get("variable", "").astype(str).str.casefold().eq(variable)
    if "tech" in pr.columns:
        tech_col = pr["tech"].fillna("").astype(str).str.casefold()
        tech_mask = tech_col.eq(tech or "")
        pr = pr[var_mask & tech_mask]
    else:
        pr = pr[var_mask]

    if pr.empty:
        return None

    def _first_match(col: str, token: str | None) -> str | None:
        if token is None or col not in pr.columns:
            return None
        m = pr[col].fillna("").astype(str).str.casefold().str.contains(token, regex=False)
        if m.any():
            return _norm_str(pr.loc[m, "proxy_region"].iloc[0])
        return None

    return (
        _first_match("applies_to_countries", country)
        or _first_match("applies_to_regions", subregion)
        or _first_match("applies_to_continents", continent)
    )


def get_val(
    df: pd.DataFrame,
    country: str,
    year,
    variable: str,
    tech: str | None = None,
    value_col: str = "value",
    proxy_rules: pd.DataFrame | None = None,
    region_map: pd.DataFrame | None = None,
    used_fallbacks: dict | None = None,
    scenario: str | None = None,
) -> float:

    proxy_rules = _DEFAULT_PROXY_RULES if proxy_rules is None else proxy_rules
    region_map  = _DEFAULT_REGION_MAP if region_map  is None else region_map

    country  = _norm_str(country) or ""
    variable = _norm_str(variable) or ""
    tech     = _norm_str(tech)
    year     = _norm_year(year)

    # -------------------------------
    # Scenario filter
    # -------------------------------
    if "scenario" in df.columns:

        if scenario is None:
            df = df[df["scenario"].isna() | (df["scenario"].astype(str).str.strip() == "")]
            if df.empty:
                msg = (
                    "No rows found for default (empty) scenario. "
                    "Provide scenario='xxx' explicitly."
                )
                logger.error(msg)
                raise ValueError(msg)

        else:
            scenario_norm = _norm_str(scenario)
            df = df[df["scenario"].astype(str).str.casefold() == scenario_norm]

            if df.empty:
                msg = f"No rows found for scenario='{scenario}'"
                logger.error(msg)
                raise ValueError(msg)

    # -------------------------------
    # Fallback chain
    # -------------------------------
    candidates: list[str | None] = [country]

    if country in region_map.index:
        candidates.append(_norm_str(region_map.at[country, "subregion"]))
        candidates.append(_norm_str(region_map.at[country, "continent"]))
    else:
        candidates += [None, None]

    candidates.append(_proxy_region(country, variable, tech, region_map, proxy_rules))
    candidates.append("world")

    # -------------------------------
    # Try in order
    # -------------------------------
    for idx, region in enumerate(candidates):

        vals = _match_series(df, region, variable, tech, year, value_col)

        if vals.empty:
            continue

        if used_fallbacks is not None and idx > 0:
            used_fallbacks[(country, variable, tech, year)] = region
            logger.info(
                f"Fallback used for {country}/{variable}/{tech}/{year}: {region}"
            )

        if len(vals) > 1:
            val = float(vals.astype(float).mean())
            logger.warning(
                f"Multiple matches for {country}, {variable}, {tech}, {year}; "
                f"returning mean={val:.4f}"
            )
            return val

        return float(vals.iloc[0])

    # -------------------------------
    # Failure
    # -------------------------------
    msg = (
        f"No match found for: Country='{country}', "
        f"Year='{year if year is not None else 'ALL'}', Var='{variable}', Tech='{tech or 'N/A'}', "
        f"Scenario='{scenario or 'default(empty)'}'"
    )
    logger.error(msg)
    raise ValueError(msg)


if __name__ == "__main__":

    print("Country norm:", _norm_str("Saudi Arabia"))

    print("Region map row for Saudi Arabia:")
    print(_DEFAULT_REGION_MAP.loc[_norm_str("Saudi Arabia")])

    fuel_gas_rules = _DEFAULT_PROXY_RULES[
        _DEFAULT_PROXY_RULES["variable"].astype(str).str.casefold().eq("fuel") &
        _DEFAULT_PROXY_RULES["tech"].astype(str).str.casefold().eq("gas")
    ]
    print("fuel/gas proxy rules:")
    print(fuel_gas_rules)

    print("_proxy_region result:")
    print(
        _proxy_region(
            country=_norm_str("Saudi Arabia"),
            variable=_norm_str("fuel"),
            tech=_norm_str("gas"),
            region_map=_DEFAULT_REGION_MAP,
            proxy_rules=_DEFAULT_PROXY_RULES,
        )
    )

    CWD = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(CWD, "..", "inputs")
    capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))

    fuel = get_val(
        capex_opex_df,
        "Saudi Arabia",
        2024,
        "fuel",
        "gas",
        "value"
    )

    print(fuel)
