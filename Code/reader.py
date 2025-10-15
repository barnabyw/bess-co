def get_val(df, country, year, variable, tech=None, type=None, value_col="value",
    proxy_rules=None,
    region_map=None,
    used_fallbacks=None,
):
    """
    Retrieve a data value for a given country, year, variable, and tech,
    with hierarchical fallback using proxy rules and regional mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns ['region', 'year', 'variable', 'tech', value_col, ...]
    country : str
        Target country name (case-insensitive)
    year : int
        Target year
    variable : str
        Variable name (e.g. 'capex', 'fuel')
    tech : str, optional
        Technology name (e.g. 'solar', 'gas')
    type : str, optional
        Additional variable subtype (optional column in df)
    value_col : str, default 'value'
        Column to extract numeric values from
    proxy_rules : pd.DataFrame, optional
        DataFrame containing proxy mappings by variable/tech (see below)
    region_map : dict, optional
        Mapping of countries to their region/continent, e.g.
        {'kenya': {'region': 'east africa', 'continent': 'africa'}}
    used_fallbacks : dict, optional
        Optional dictionary to log which fallback region was used

    Returns
    -------
    float
        The retrieved (or proxied) value
    """

    # --- Normalise inputs ---
    country = country.strip().lower()
    variable = variable.strip().lower()
    tech = tech.lower() if tech else None
    type = type.lower() if type else None

    # --- Helper: mask builder ---
    def build_mask(target_region):
        mask = (
            df["region"].str.lower().eq(target_region)
            & df["year"].eq(year)
            & df["variable"].str.lower().eq(variable)
        )
        if tech and "tech" in df.columns:
            mask &= df["tech"].str.lower().eq(tech)
        if type and "type" in df.columns:
            mask &= df["type"].str.lower().eq(type)
        return mask

    # --- Helper: find proxy region from rules ---
    def get_proxy_region(country, variable, tech, proxy_rules, region_map):
        if proxy_rules is None:
            return None
        rules = proxy_rules[
            (proxy_rules["variable"].str.lower() == variable)
            & (proxy_rules["tech"].str.lower() == (tech or ""))
        ]

        # Get region and continent for the country
        region = region_map.get(country, {}).get("region") if region_map else None
        continent = region_map.get(country, {}).get("continent") if region_map else None

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

        return None  # no proxy match found

    # --- Step 1: Try direct country match ---
    subset = df.loc[build_mask(country)]

    # --- Step 2: Proxy rule fallback ---
    if subset.empty and proxy_rules is not None:
        proxy_region = get_proxy_region(country, variable, tech, proxy_rules, region_map)
        if proxy_region:
            subset = df.loc[build_mask(proxy_region)]
            if not subset.empty and used_fallbacks is not None:
                used_fallbacks[(country, variable, tech, year)] = proxy_region
                print(f"No data for {country.title()} ({year}, {variable}, {tech or ''}); using proxy '{proxy_region.title()}'")
        else:
            print(f"No proxy rule matched for {country.title()} ({year}, {variable}, {tech or ''})")

    # --- Step 3: Global fallback ---
    if subset.empty:
        subset = df.loc[build_mask("world")]
        if not subset.empty:
            print(f"No data for {country.title()} ({year}, {variable}, {tech or ''}); using 'World'")
            if used_fallbacks is not None:
                used_fallbacks[(country, variable, tech, year)] = "world"
        else:
            raise ValueError(f"No match found for {country}, {year}, {variable}, {tech or ''}, {type or ''}")

    # --- Step 4: Handle duplicates ---
    if len(subset) > 1:
        mean_val = subset[value_col].mean()
        print(f"Warning: {len(subset)} matches found for {country}, {year}, {variable}, {tech or ''}; returning mean")
        return mean_val

    # --- Step 5: Return result ---
    return float(subset.iloc[0][value_col])
