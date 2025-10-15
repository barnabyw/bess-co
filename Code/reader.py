def get_val(df, country, year, variable, tech=None, type=None, value_col="value"):

    #normalise
    country = country.strip().lower()
    variable = variable.strip().lower()
    tech = tech.lower() if tech else None
    type = type.lower() if type else None

    def build_mask(target_country):
        # Base filter
        mask = (
            df["region"].str.lower().eq(target_country)
            & df["year"].eq(year)
            & df["variable"].str.lower().eq(variable)
        )

        if tech and "tech" in df.columns:
            mask &= df["tech"].str.lower().eq(tech)
        if type and "type" in df.columns:
            mask &= df["type"].str.lower().eq(type)

        return mask

    subset = df.loc[build_mask(country)]

    if subset.empty:
        subset = df.loc[build_mask("world")]
        if not subset.empty:
            print(f"No data for country {country.title()} ({year}, {variable}, {tech or ''}), using 'World' instead")
        else:
            raise ValueError(f"No match found for {country}, {year}, {variable}, {tech or ''}, {type or ''}")

    #If multiple matches return mean and warn
    if len(subset) > 1:
        mean_val = subset[value_col].mean()
        print(f"Warning... {len(subset)} matches found for {country}, {year}, {variable}, {tech or ''}, {type or ''}")
        return mean_val

    return float(subset.iloc[0][value_col])