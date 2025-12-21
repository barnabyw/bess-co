import pandas as pd
from pathlib import Path

def ingest_wide_excel_to_long(
    file_path: str | Path,
    sheet_name: str,
    tech: str,
    units: str,
    variable: str,
    scenario: str | None,
    type_: str,
    money: str,
    money_year,
    source: str,
) -> pd.DataFrame:
    """
    Read a wide-format Excel sheet (Year + region columns),
    convert to long format, and attach metadata for database ingestion.
    """

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Normalise column names
    df.columns = df.columns.astype(str).str.strip()

    if "Year" not in df.columns:
        raise ValueError("Input sheet must contain a 'Year' column.")

    # -----------------------------
    # Unpivot to long format
    # -----------------------------
    long_df = df.melt(
        id_vars="Year",
        var_name="region",
        value_name="value",
    )

    # Drop missing values
    long_df["value"] = round(long_df["value"],1)

    # -----------------------------
    # Add metadata columns
    # -----------------------------
    long_df = long_df.rename(columns={"Year": "year"})

    long_df["region"] = long_df["region"].astype(str).str.strip()
    long_df["tech"] = tech
    long_df["units"] = units
    long_df["variable"] = variable
    long_df["scenario"] = scenario or ""
    long_df["type"] = type_
    long_df["money"] = money
    long_df["money year"] = money_year
    long_df["source"] = source

    # -----------------------------
    # Column order (important)
    # -----------------------------
    long_df = long_df[
        [
            "year",
            "value",
            "region",
            "tech",
            "units",
            "variable",
            "scenario",
            "type",
            "money",
            "money year",
            "source",
        ]
    ]

    return long_df


df_new = ingest_wide_excel_to_long(
    file_path=r"C:\Users\barna\OneDrive\Documents\Solar_BESS\inputs\workings\filling_data.xlsx",
    sheet_name="bess_energy",
    tech="BESS",
    units="kWh",
    variable="capex_e",
    scenario=None,
    type_="historic",
    money="USD",
    money_year=2024,
    source="IRENA/Bloomberg/Modo",
)

# Append to database
db_path = Path(r"C:\Users\barna\OneDrive\Documents\Solar_BESS\inputs\db.csv")

if db_path.exists():
    df_existing = pd.read_csv(db_path)
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_out = df_new

df_out.to_csv(db_path, index=False)
