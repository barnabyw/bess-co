from pathlib import Path
import pandas as pd
import logging
logger = logging.getLogger(__name__)

# --- Load default mappings (quietly tolerate absence) ---
_DATA_DIR = Path(__file__).resolve().parent / ".." / "mappings"

_DEFAULT_PROXY_RULES = pd.read_csv(_DATA_DIR / "proxy_rules.csv")
_DEFAULT_REGION_MAP = pd.read_csv(_DATA_DIR / "region_map.csv")
_DEFAULT_REGION_MAP["country"] = _DEFAULT_REGION_MAP["country"].str.strip().str.casefold()
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
    """Return proxy region from rules, else None (we already try subregion/continent earlier)."""
    if proxy_rules is None or proxy_rules.empty:
        return None

    # Look up subregion/continent for rule matching (best-effort)
    subregion = continent = None
    if country in region_map.index:
        subregion = _norm_str(region_map.at[country, "subregion"])
        continent = _norm_str(region_map.at[country, "continent"])

    pr = proxy_rules.copy()

    # filter rules by variable + (tech or blank)
    var_mask = pr.get("variable", "").astype(str).str.casefold().eq(variable)
    if "tech" in pr.columns:
        tech_col = pr["tech"].fillna("").astype(str).str.casefold()
        tech_mask = tech_col.eq(tech or "")
        pr = pr[var_mask & tech_mask]
    else:
        pr = pr[var_mask]

    if pr.empty:
        return None

    # prefer country match, then subregion, then continent
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
    audit_log: list | None = None,
    audit_context: dict | None = None,
) -> float:

    """
    Hierarchy: country → subregion → continent → proxy (rules) → world.
    - Case-insensitive matching on strings
    - Year can be int/'all'/'*'/None
    - If multiple matches: returns mean and prints a warning
    - Records fallbacks into `used_fallbacks` dict if provided
    """
    proxy_rules = _DEFAULT_PROXY_RULES if proxy_rules is None else proxy_rules
    region_map = _DEFAULT_REGION_MAP if region_map is None else region_map

    country = _norm_str(country) or ""
    variable = _norm_str(variable) or ""
    tech = _norm_str(tech)
    year = _norm_year(year)

    # Build candidates: country → subregion → continent → proxy → world
    candidates: list[str | None] = [country]

    if country in region_map.index:
        candidates.append(_norm_str(region_map.at[country, "subregion"]))
        candidates.append(_norm_str(region_map.at[country, "continent"]))
    else:
        candidates += [None, None]

    candidates.append(_proxy_region(country, variable, tech, region_map, proxy_rules))
    candidates.append("world")

    # Try each candidate in order
    for idx, region in enumerate(candidates):
        vals = _match_series(df, region, variable, tech, year, value_col)
        if vals.empty:
            continue

        # Record fallback if not the first (country)
        if used_fallbacks is not None and idx > 0:
            used_fallbacks[(country, variable, tech, year)] = region

        # --- AUDIT LOGGING --------------------------------------------------
        if audit_log is not None:
            audit_entry = {
                "query_country": country,
                "query_variable": variable,
                "query_tech": tech,
                "query_year": year,
                "region_used": region,
                "df_indices": list(vals.index),
                "df_values": vals.tolist(),
            }
            if audit_context:
                audit_entry.update(audit_context)
            audit_log.append(audit_entry)
        # --------------------------------------------------------------------

        if len(vals) > 1:
            val = float(vals.astype(float).mean())
            logger.warning(
                "Multiple (%d) matches for Country='%s', Year='%s', Var='%s', Tech='%s'; "
                "returning mean=%f.",
                len(vals),
                country,
                year if year is not None else "ALL",
                variable,
                tech or "N/A",
                val,
            )
            return val

        return float(vals.iloc[0])

    msg = (
        f"FATAL: No match found for: Country='{country}', "
        f"Year='{year if year is not None else 'ALL'}', Var='{variable}', Tech='{tech or 'N/A'}'"
    )
    logger.error(msg)
    raise ValueError(msg)

if __name__ == "__main__":
    import os

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
    print(_proxy_region(
        country=_norm_str("Saudi Arabia"),
        variable=_norm_str("fuel"),
        tech=_norm_str("gas"),
        region_map=_DEFAULT_REGION_MAP,
        proxy_rules=_DEFAULT_PROXY_RULES,
    ))

    CWD = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(CWD, "..", "inputs")
    capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))

    fuel = get_val(
    capex_opex_df,
    "Saudi Arabia",
    2024,
    "fuel",
    "gas",
    "value")

    print(fuel)