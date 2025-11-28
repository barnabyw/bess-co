import numpy as np
import pandas as pd
import os

# === Configuration ===
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")


def fit_learning_curve(path, cum_col="gwh capacity", cost_col="$/kWh", sheet_name='battery_kwh'):
    """
    Fits a standard experience curve:
        cost = A * (cumulative_capacity)^(-lambda)

    Returns:
        dict with lambda, learning_rate, A, predict_cost, and cleaned df
    """

    # ---- Read Excel ----
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=1)

    # ---- Clean data ----
    required_cols = ["Year", cum_col, cost_col]
    df = df[required_cols].dropna()
    df = df[df[cum_col] > 0]

    # Extract variables
    X = np.log(df[cum_col].values.astype(float))
    y = np.log(df[cost_col].values.astype(float))

    # Fit log-log linear regression: log(cost) = a + b log(cum)
    b, a = np.polyfit(X, y, 1)

    lam = -b
    learning_rate = 1 - 2**(-lam)
    A = np.exp(a)

    def predict_cost(cum_capacity):
        cum_capacity = np.asarray(cum_capacity, dtype=float)
        return A * cum_capacity ** (-lam)

    # Pretty print
    print("\n================ Learning Curve Fit ================")
    print(f"λ (learning parameter):      {lam:.4f}")
    print(f"Learning rate per doubling:  {learning_rate*100:.1f}%")
    print(f"A (cost at 1 unit cum cap):  {A:.2f} $/kWh")
    print("Model: cost = A * (cum)^(-λ)")
    print("====================================================\n")

    # Clean df for output
    df_clean = df.rename(columns={cum_col: "installed_cap_gwh", cost_col: "bess_capex_kwh", "Year": "year"})
    df_clean = df_clean.sort_values("year").reset_index(drop=True)

    return {
        "lambda": lam,
        "learning_rate": learning_rate,
        "A": A,
        "predict_cost": predict_cost,
        "df": df_clean
    }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    input_file = os.path.join(INPUT_PATH, "bess_installs_capex.xlsx")
    result = fit_learning_curve(input_file)

    df_hist = result["df"]

    # === Create future projection to 2040 using CAGR ===
    start_year = int(df_hist["year"].iloc[-1])
    last_cum = df_hist["installed_cap_gwh"].iloc[-1]

    end_year = 2040
    years_future = np.arange(start_year + 1, end_year + 1)

    CAGR = 0.20   # <-- set CAGR here (20% example)

    projected_cum = last_cum * (1 + CAGR) ** (years_future - start_year)
    projected_cost = result["predict_cost"](projected_cum)

    df_future = pd.DataFrame({
        "year": years_future,
        "installed_cap_gwh": np.round(projected_cum,0),
        "bess_capex_kwh": np.round(projected_cost, 1)
    })

    # === Combine past + future ===
    df_all = pd.concat([df_hist, df_future], ignore_index=True)

    # === Save ONE CSV ===
    output_path = os.path.join(INPUT_PATH, "bess_learning_curve.csv")
    df_all.to_csv(output_path, index=False)

    print(f"\nSaved historical + projected learning curve to:\n  {output_path}\n")
