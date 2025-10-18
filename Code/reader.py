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
        year: int,
        variable: str,
        tech: str = None,
        param_type: str = None,  # Renamed 'type' to avoid conflict with Python's built-in
        value_col: str = "value",
        proxy_rules: pd.DataFrame = None,
        region_map: pd.DataFrame = None,
        used_fallbacks: dict = None,
) -> float:
    """
    Retrieve a data value with hierarchical fallback using proxy rules & region map.

    Args:
        df (pd.DataFrame): The main data DataFrame.
        country (str): The target country.
        year (int): The target year.
        variable (str): The variable to look up (e.g., 'capex', 'fuel').
        tech (str, optional): The technology (e.g., 'Solar', 'Gas'). Defaults to None.
        param_type (str, optional): The parameter type (e.g., 'fixed', 'variable'). Defaults to None.
        value_col (str, optional): The name of the column containing the value. Defaults to "value".
        proxy_rules (pd.DataFrame, optional): DataFrame with proxy rules. Defaults to loaded default.
        region_map (pd.DataFrame, optional): DataFrame mapping countries to regions. Defaults to loaded default.
        used_fallbacks (dict, optional): A dictionary to record when fallbacks are used.

    Returns:
        float: The retrieved data value.

    Raises:
        ValueError: If no value can be found after all fallbacks.
    """
    # Use defaults if not provided
    proxy_rules = _DEFAULT_PROXY_RULES if proxy_rules is None else proxy_rules
    region_map = _DEFAULT_REGION_MAP if region_map is None else region_map

    # --- 1. Normalize all inputs for consistent matching ---
    country = country.strip().lower()
    variable = variable.strip().lower()
    tech = tech.strip().lower() if tech else None
    param_type = param_type.strip().lower() if param_type else None

    # --- Helper to perform the actual lookup ---
    def find_value(target_region: str):
        # Build a boolean mask to filter the DataFrame
        mask = (
                (df["region"].str.lower() == target_region) &
                (df["year"] == year) &
                (df["variable"].str.lower() == variable)
        )
        # Conditionally add filters for tech and type if they are provided
        if tech and "tech" in df.columns:
            mask &= df["tech"].str.lower() == tech
        if param_type and "type" in df.columns:
            mask &= df["type"].str.lower() == param_type

        return df[mask]

    # --- Helper to find a proxy region from the rules ---
    def get_proxy_region():
        if proxy_rules.empty or region_map.empty:
            return None

        try:
            country_info = region_map.loc[country]
            subregion = country_info['subregion'].lower()
            continent = country_info['continent'].lower()
        except KeyError:  # Country not in region map
            subregion, continent = None, None

        # Filter rules for the specific variable and tech
        rule_mask = (proxy_rules["variable"].str.lower() == variable)
        # Handle cases where tech might be None or not applicable
        if tech:
            rule_mask &= (proxy_rules["tech"].fillna('').str.lower() == tech)

        filtered_rules = proxy_rules[rule_mask]

        for _, rule in filtered_rules.iterrows():
            # Check for country, then region, then continent match
            if country in str(rule.get("applies_to_countries", "")).lower():
                return rule["proxy_region"].lower()
            if subregion and subregion in str(rule.get("applies_to_regions", "")).lower():
                return rule["proxy_region"].lower()
            if continent and continent in str(rule.get("applies_to_continents", "")).lower():
                return rule["proxy_region"].lower()
        return None

    # --- 2. Attempt lookups in hierarchical order ---
    # Level 1: Direct country match
    subset = find_value(country)

    # Level 2: Proxy region fallback
    if subset.empty:
        proxy = get_proxy_region()
        if proxy:
            subset = find_value(proxy)
            if not subset.empty and used_fallbacks is not None:
                used_fallbacks[(country, variable, tech, year)] = proxy
                print(
                    f"INFO: No direct data for {country.title()} ({variable}, {tech or ''}, {year}). Using proxy '{proxy.title()}'.")

    # Level 3: Global 'world' fallback
    if subset.empty:
        subset = find_value("world")
        if not subset.empty and used_fallbacks is not None:
            used_fallbacks[(country, variable, tech, year)] = "world"
            print(
                f"INFO: No direct/proxy data for {country.title()} ({variable}, {tech or ''}, {year}). Using 'World' default.")

    # --- 3. Process the result or raise an error ---
    if subset.empty:
        raise ValueError(
            f"FATAL: No match found for: Country='{country}', Year='{year}', Var='{variable}', Tech='{tech or 'N/A'}'")

    if len(subset) > 1:
        val = subset[value_col].mean()
        print(f"WARNING: Found {len(subset)} matches for the query. Returning the mean value: {val}.")
        return val

    return float(subset.iloc[0][value_col])