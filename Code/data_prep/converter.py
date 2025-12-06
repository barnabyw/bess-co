import pandas as pd
import os

from pathlib import Path
from Code.archive.assumptions import base_path

# === CONFIG ===
input_path = os.path.join(base_path,"inputs")
input_file = os.path.join(input_path,"capex_opex.xlsx")
input_file2 = os.path.join(input_path,"capex_opex_2.xlsx")

# === OUTPUT PATH (MAIN INPUT) ===
CWD = Path(__file__).resolve().parent         # folder containing this script
outer_folder = CWD.parent                     # one level up
output_path = os.path.join(outer_folder, "..", "inputs")
output_file = os.path.join(output_path,"capex_opex_converted.xlsx")

log_file = os.path.join(output_path,"conversion_log.csv")

# === LOAD DATA ===
capex_opex_df = pd.read_excel(input_file2)
other_df = pd.read_excel(input_file, sheet_name="other")
deflators = pd.read_excel(input_file, sheet_name="deflators")
exchange = pd.read_excel(input_file, sheet_name="exchange_rates")
unit_df = pd.read_excel(input_file, sheet_name="unit_conversion")

# Set target
TARGET_YEAR = 2025
TARGET_CURRENCY = "USD"

# --- Create lookups ---
deflators = deflators.set_index("year")["USD_deflator"].to_dict()
exchange = exchange.set_index("year").to_dict(orient="index")

def _norm(s):
    return str(s).strip().lower() if pd.notna(s) else ""

unit_df["context"] = unit_df["context"].fillna("").map(_norm)
unit_df["from_unit_norm"] = unit_df["from_unit"].map(_norm)
unit_df["to_unit_norm"] = unit_df["to_unit"].map(_norm)

# prefer exact tech context; fall back to "general"
unit_conversions = {}
for _, r in unit_df.iterrows():
    ctx = r["context"] or "general"
    key = (r["from_unit_norm"], ctx)
    unit_conversions[key] = {
        "to_unit": r["to_unit"],
        "to_unit_norm": r["to_unit_norm"],
        "multiplier": float(r["multiplier"]),
        "context": ctx
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

def get_unit_conversion(from_unit, tech):
    fu = (from_unit or "").strip().lower()
    ctx = (tech or "").strip().lower()

    # exact tech context
    conv = unit_conversions.get((fu, ctx))
    if conv:
        return conv
    # general fallback
    return unit_conversions.get((fu, "general"))

# === INITIALISE LOG ===
conversion_log = []

# === MAIN CONVERSION ===
def convert_row(row):
    # ---- inputs / originals
    value = row.get("value")
    if pd.isna(value):
        return row

    orig_value = value
    orig_money = str(row.get("money", "USD")).upper().strip()
    orig_year  = row.get("money year", TARGET_YEAR)
    orig_units = str(row.get("units", "")).strip()
    tech       = str(row.get("tech", "")).strip()

    # 1) FX to USD (same-year)
    fx_rate = get_exchange_rate(orig_money, orig_year)
    value = value * fx_rate

    # 2) Inflate to target-year USD
    deflator = get_deflator(orig_year)
    value = value * deflator

    # 3) Unit conversion (only if a rule exists AND units differ)
    conv = get_unit_conversion(orig_units, tech)
    if conv:
        value *= conv["multiplier"]
        new_units = conv["to_unit"]
        ctx_used = conv["context"]
        unit_mult = conv["multiplier"]
    else:
        new_units = orig_units
        ctx_used = "none"
        unit_mult = 1.0

    # ---- update row
    row["value"] = value
    row["money"] = TARGET_CURRENCY
    row["money year"] = TARGET_YEAR
    row["units"] = new_units

    # ---- log
    conversion_log.append({
        "tech": tech,
        "variable": row.get("variable"),
        "from_currency": orig_money,
        "to_currency": TARGET_CURRENCY,
        "currency_year": orig_year,
        "fx_rate": fx_rate,
        "deflator": deflator,
        "from_unit": orig_units,
        "to_unit": new_units,
        "unit_multiplier": unit_mult,
        "context_used": ctx_used,
        "value_before": orig_value,
        "value_after": value,
        "note": "" if conv else f"No unit rule for '{orig_units}' with tech '{tech}', kept units."
    })

    return row

# === APPLY CONVERSIONS ===
df_converted = capex_opex_df.apply(convert_row, axis=1)

# === ADD FIXED VALUES (WACC, LIFETIME, ETC.) ===
full_df = pd.concat([df_converted, other_df], ignore_index=True)

# === SAVE OUTPUTS ===
full_df.to_excel(output_file, index=False)
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