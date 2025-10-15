import pandas as pd
import os

from assumptions import base_path

# === CONFIG ===
input_path = os.path.join(base_path,"inputs")
input_file = os.path.join(input_path,"capex_opex.xlsx")
output_file = os.path.join(input_path,"capex_opex_converted_2025USD.xlsx")

log_file = os.path.join(input_path,"conversion_log.csv")

# === LOAD DATA ===
df = pd.read_excel(input_file, sheet_name="capex_opex")
deflators = pd.read_excel(input_file, sheet_name="deflators")
exchange = pd.read_excel(input_file, sheet_name="exchange_rates")
unit_df = pd.read_excel(input_file, sheet_name="unit_conversion")


# Set target
TARGET_YEAR = 2025
TARGET_CURRENCY = "USD"

# --- Create lookups ---
deflators = deflators.set_index("year")["USD_deflator"].to_dict()
exchange = exchange.set_index("year").to_dict(orient="index")
unit_conversions = {
    (row["from_unit"], str(row.get("context", "")).lower() or "general"): {
        "to_unit": row["to_unit"],
        "multiplier": float(row["multiplier"]),
        "context": str(row.get("context", "general"))
    }
    for _, row in unit_df.iterrows()
}

# === HELPER FUNCTIONS ===
def get_deflator(year):
    """Return multiplier to bring given-year USD → 2025 USD."""
    try:
        year = int(year)
    except Exception:
        return 1.0
    return deflators.get(year, 1.0)

def get_exchange_rate(currency, year):
    """Return FX rate from given currency → USD for the given year."""
    try:
        year = int(year)
    except Exception:
        year = TARGET_YEAR
    if year not in exchange:
        valid_years = sorted(exchange.keys())
        year = max([y for y in valid_years if y <= year], default=TARGET_YEAR)
    rates = exchange[year]
    if currency == "USD":
        return 1.0
    col = f"{currency}_to_USD"
    return rates.get(col, 1.0)

def get_unit_conversion(from_unit, context="general"):
    """Return multiplier and new unit based on unit and tech/fuel context."""
    context = context.lower()
    return (
        unit_conversions.get((from_unit, context))
        or unit_conversions.get((from_unit, "general"))
    )

# === INITIALISE LOG ===
conversion_log = []

# === MAIN CONVERSION ===
def convert_row(row):
    money = str(row.get("money", "USD")).upper().strip()
    money_year = row.get("money year", TARGET_YEAR)
    value = row["value"]
    if pd.isna(value):
        return row

    original_value = value
    original_money = money
    original_year = money_year
    original_units = str(row.get("units", "")).strip()

    # 1️⃣ Currency conversion to USD for that year
    fx_rate = get_exchange_rate(money, money_year)
    value_usd = value * fx_rate

    # 2️⃣ Inflation adjustment to 2025 USD
    deflator = get_deflator(money_year)
    value_2025usd = value_usd * deflator

    # 3️⃣ Unit conversion
    tech_context = f"{row.get('tech', '')} {row.get('variable', '')}".lower()
    conv = get_unit_conversion(original_units, tech_context)
    if conv:
        value_2025usd *= conv["multiplier"]
        new_units = conv["to_unit"]
        conv_context = conv["context"]
    else:
        new_units = original_units
        conv_context = "none"

    # Update row
    row["value"] = value_2025usd
    row["money"] = TARGET_CURRENCY
    row["money year"] = TARGET_YEAR
    row["units"] = new_units

    # Append to log
    conversion_log.append({
        "tech": row.get("tech"),
        "variable": row.get("variable"),
        "from_currency": original_money,
        "to_currency": TARGET_CURRENCY,
        "currency_year": original_year,
        "fx_rate": fx_rate,
        "deflator": deflator,
        "from_unit": original_units,
        "to_unit": new_units,
        "unit_multiplier": conv["multiplier"] if conv else 1.0,
        "context_used": conv_context,
        "value_before": original_value,
        "value_after": value_2025usd
    })

    return row

# === APPLY CONVERSIONS ===
df_converted = df.apply(convert_row, axis=1)

# === SAVE OUTPUTS ===
df_converted.to_excel(output_file, index=False)
log_df = pd.DataFrame(conversion_log)
log_df.to_csv(log_file, index=False)

# === PRINT SUMMARY ===
print("\n✅ Conversion complete.")
print(f"Saved converted data → {os.path.abspath(output_file)}")
print(f"Saved conversion log → {os.path.abspath(log_file)}")

# Summarise key actions to console
summary = log_df.groupby(["from_currency", "currency_year"]).size().reset_index(name="rows")
for _, r in summary.iterrows():
    print(f"• {int(r['rows'])} rows converted from {r['from_currency']} {r['currency_year']} → USD {TARGET_YEAR}")

unit_summary = log_df.groupby(["from_unit", "to_unit"]).size().reset_index(name="rows")
for _, r in unit_summary.iterrows():
    if r["from_unit"] != r["to_unit"]:
        print(f"• {int(r['rows'])} unit conversions: {r['from_unit']} → {r['to_unit']}")