import pandas as pd
import os
import numpy as np

# --- Load and Prepare Default Mappings ---
try:
    # Project_Root/
    #  |- Code/
    #  |   |- get_val_module.py
    #  |- mappings/
    #      |- proxy_rules.csv
    #      |- region_map.csv
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "mappings")

    _DEFAULT_PROXY_RULES = pd.read_csv(os.path.join(DATA_DIR, "proxy_rules.csv"))

    # Keep region_map as a DataFrame indexed by country for easier lookups
    _DEFAULT_REGION_MAP = pd.read_csv(os.path.join(DATA_DIR, "region_map.csv"))
    _DEFAULT_REGION_MAP['country'] = _DEFAULT_REGION_MAP['country'].str.lower()
    _DEFAULT_REGION_MAP = _DEFAULT_REGION_MAP.set_index('country')

except FileNotFoundError as e:
    print(f"Warning: Could not load default mapping files. Error: {e}")
    _DEFAULT_PROXY_RULES = pd.DataFrame()
    _DEFAULT_REGION_MAP = pd.DataFrame()

# --- Main Data Retrieval Function ---
def get_val(
        df: pd.DataFrame,
        country: str,
        year,
        variable: str,
        tech: str = None,
        value_col: str = "value",
        proxy_rules: pd.DataFrame = None,
        region_map: pd.DataFrame = None,
        used_fallbacks: dict = None,
) -> float:
    # defaults
    proxy_rules = _DEFAULT_PROXY_RULES if proxy_rules is None else proxy_rules
    region_map = _DEFAULT_REGION_MAP if region_map is None else region_map

    # --- normalize inputs ---
    country = (country or "").strip().lower()
    variable = (variable or "").strip().lower()
    tech = (tech or "").strip().lower() or None

    # year: accept int, "2024", None, "all", "*"
    def _normalize_year(y):
        if y is None:
            return None
        if isinstance(y, str) and y.strip().lower() in {"all", "any", "*"}:
            return None
        try:
            return int(y)
        except Exception:
            return None

    year = _normalize_year(year)

    # ensure df.year is numeric once (safe even if already numeric)
    if "year" in df.columns:
        try:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
        except Exception:
            pass

    # --- lookup helper ---
    def find_value(target_region: str):
        if not isinstance(target_region, str):
            return df.iloc[0:0]  # empty
        mask = (df["region"].str.lower() == target_region.strip().lower()) \
               & (df["variable"].str.lower() == variable)
        if tech and "tech" in df.columns:
            mask &= (df["tech"].str.lower() == tech)
        if year is not None and "year" in df.columns:
            mask &= (df["year"] == year)
        return df[mask]

    # --- proxy helper (keeps your rules; falls back to subregion if rules don't match) ---
    def get_proxy_region():
        # try rules first
        if proxy_rules is not None and not proxy_rules.empty:
            try:
                subregion = str(region_map.loc[country, "subregion"]).lower()
                continent = str(region_map.loc[country, "continent"]).lower()
            except Exception:
                subregion = continent = None

            rm = (proxy_rules["variable"].str.lower() == variable)
            if tech:
                rm &= (proxy_rules["tech"].fillna("").str.lower() == tech)
            pr = proxy_rules[rm]

            for _, rule in pr.iterrows():
                if country in str(rule.get("applies_to_countries", "")).lower():
                    return str(rule["proxy_region"]).lower()
                if subregion and subregion in str(rule.get("applies_to_regions", "")).lower():
                    return str(rule["proxy_region"]).lower()
                if continent and continent in str(rule.get("applies_to_continents", "")).lower():
                    return str(rule["proxy_region"]).lower()

        # if no rule hit, just use the country’s subregion as proxy
        try:
            return str(region_map.loc[country, "subregion"]).strip().lower()
        except Exception:
            return None

    # ========== HIERARCHY ==========
    # 1) Country (region==country)
    subset = find_value(country)

    # 2) Country's subregion (direct, before explicit rules/world)
    if subset.empty and region_map is not None and not region_map.empty:
        try:
            sr = str(region_map.loc[country, "subregion"]).strip().lower()
            subset = find_value(sr)
            if not subset.empty and used_fallbacks is not None:
                used_fallbacks[(country, variable, tech, year)] = sr
                print(f"INFO: No country data for {country.title()} → using subregion '{sr.title()}'.")
        except KeyError:
            pass

    # 3) Proxy region (rules or subregion as default from helper)
    if subset.empty:
        proxy = get_proxy_region()
        if proxy:
            subset = find_value(proxy)
            if not subset.empty and used_fallbacks is not None:
                used_fallbacks[(country, variable, tech, year)] = proxy
                print(f"INFO: Using proxy region '{proxy.title()}' for {country.title()}.")

    # 4) World
    if subset.empty:
        subset = find_value("world")
        if not subset.empty and used_fallbacks is not None:
            used_fallbacks[(country, variable, tech, year)] = "world"
            print(f"INFO: Falling back to World for {country.title()}.")

    # ========== RESULT ==========
    if subset.empty:
        raise ValueError(
            f"FATAL: No match found for: Country='{country}', Year='{year if year is not None else 'ALL'}', Var='{variable}', Tech='{tech or 'N/A'}'"
        )

    if len(subset) > 1:
        val = float(subset[value_col].mean())
        print(f"WARNING: {len(subset)} matches; returning mean={val}.")
        return val

    return float(subset.iloc[0][value_col])
