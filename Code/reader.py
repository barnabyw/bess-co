# get_val_module.py
import pandas as pd
import os

# --- Load defaults ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "mappings")

# Read your default mapping files
try:
    _DEFAULT_PROXY_RULES = pd.read_csv(os.path.join(DATA_DIR, "proxy_rules.csv"))
    _DEFAULT_REGION_MAP = pd.read_csv(os.path.join(DATA_DIR, "region_map.csv"))
    _DEFAULT_REGION_MAP = (
        _DEFAULT_REGION_MAP
        .assign(country=lambda df: df["country"].str.lower(),
                region=lambda df: df["subregion"].str.lower(),
                continent=lambda df: df["continent"].str.lower())
        .set_index("country")
        .apply(lambda x: x.to_dict(), axis=1)
        .to_dict()
    )
except FileNotFoundError:
    print("Warning: Default proxy_rules.csv or region_map.csv not found.")
    _DEFAULT_PROXY_RULES = pd.DataFrame()
    _DEFAULT_REGION_MAP = {}

# --- Main function ---
def get_val(
    df,
    country,
    year,
    variable,
    tech=None,
    type=None,
    value_col="value",
    proxy_rules=None,
    region_map=None,
    used_fallbacks=None,
):
    """
    Retrieve a data value with hierarchical fallback using proxy rules & region map.
    """

    # Use defaults if not provided
    if proxy_rules is None:
        proxy_rules = _DEFAULT_PROXY_RULES
    if region_map is None:
        region_map = _DEFAULT_REGION_MAP

    # Normalise
    country = country.strip().lower()
    variable = variable.strip().lower()
    tech = tech.lower() if tech else None
    type = type.lower() if type else None

    def build_mask(target):
        mask = (
            df["region"].str.lower().eq(target)
            & df["year"].eq(year)
            & df["variable"].str.lower().eq(variable)
        )
        if tech and "tech" in df.columns:
            mask &= df["tech"].str.lower().eq(tech)
        if type and "type" in df.columns:
            mask &= df["type"].str.lower().eq(type)
        return mask

    # --- Proxy lookup helper ---
    def get_proxy_region(country, variable, tech, proxy_rules, region_map):
        if proxy_rules.empty:
            return None
        rules = proxy_rules[
            (proxy_rules["variable"].str.lower() == variable)
            & (proxy_rules["tech"].str.lower() == (tech or ""))
        ]
        region = region_map.get(country, {}).get("region")
        continent = region_map.get(country, {}).get("continent")

        for _, rule in rules.iterrows():
            continents = [x.strip().lower() for x in str(rule.get("applies_to_continents", "")).split(",") if x]
            regions = [x.strip().lower() for x in str(rule.get("applies_to_regions", "")).split(",") if x]
            countries = [x.strip().lower() for x in str(rule.get("applies_to_countries", "")).split(",") if x]
            if (
                (country in countries)
                or (region and region in regions)
                or (continent and continent in continents)
            ):
                return rule["proxy_region"].lower()
        return None

    # --- Step 1: Try country ---
    subset = df.loc[build_mask(country)]

    # --- Step 2: Proxy fallback ---
    if subset.empty:
        proxy = get_proxy_region(country, variable, tech, proxy_rules, region_map)
        if proxy:
            subset = df.loc[build_mask(proxy)]
            if not subset.empty:
                if used_fallbacks is not None:
                    used_fallbacks[(country, variable, tech, year)] = proxy
                print(f"No data for {country.title()} ({year}, {variable}, {tech or ''}); using proxy '{proxy.title()}'")

    # --- Step 3: Global fallback ---
    if subset.empty:
        subset = df.loc[build_mask("world")]
        if not subset.empty:
            if used_fallbacks is not None:
                used_fallbacks[(country, variable, tech, year)] = "world"
            print(f"No data for {country.title()} ({year}, {variable}, {tech or ''}); using 'World'")
        else:
            raise ValueError(f"No match found for {country}, {year}, {variable}, {tech or ''}")

    if len(subset) > 1:
        val = subset[value_col].mean()
        print(f"Warning: {len(subset)} matches found for {country}, {year}, {variable}, {tech or ''}")
        return val

    return float(subset.iloc[0][value_col])

def reload_defaults():
    global _DEFAULT_PROXY_RULES, _DEFAULT_REGION_MAP
    _DEFAULT_PROXY_RULES = pd.read_csv(os.path.join(DATA_DIR, "proxy_rules.csv"))
    _DEFAULT_REGION_MAP = pd.read_csv(os.path.join(DATA_DIR, "region_map.csv"))
    print("Default proxy_rules and region_map reloaded.")
