import numpy as np
import pandas as pd
import os

CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")

# ============================================================
# 1. Fit YEAR → CUMULATIVE CAPACITY curve (growth model)
# ============================================================
def fit_capacity_growth_curve(path, cum_col="gwh capacity", sheet_name="battery_kwh"):
    """
    Fits exponential growth:
        cum(t) = C0 * exp(g * (t - t0))
    or equivalently:
        log(cum) = a + g * year

    Returns:
        - growth_rate (g)
        - predict_capacity(year)
        - cleaned df
    """

    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=1)
    df = df[["Year", cum_col]].dropna()
    df = df[df[cum_col] > 0]
    df = df.rename(columns={cum_col: "installed_cap_gwh", "Year": "year"})
    df = df.sort_values("year").reset_index(drop=True)

    years = df["year"].values.astype(float)
    log_cap = np.log(df["installed_cap_gwh"].values.astype(float))

    # simple regression
    g, a = np.polyfit(years, log_cap, 1)

    def predict_capacity(year):
        year = np.asarray(year, dtype=float)
        return np.exp(a + g * year)

    print("\n============== Capacity Growth Fit ==============")
    print(f"Growth rate (g):         {g:.4f}  (≈ {100*(np.exp(g)-1):.1f}% CAGR)")
    print("Model: cum = exp(a + g * year)")
    print("=================================================\n")

    return {
        "growth_rate": g,
        "cagr": np.exp(g) - 1,
        "predict_capacity": predict_capacity,
        "df": df,
    }


# ============================================================
# 2. Fit COST → CUMULATIVE CAPACITY learning curve
# ============================================================
def fit_learning_curve(path, cum_col="gwh capacity", cost_col="$/kWh", sheet_name='battery_kwh'):
    """
    Fits experience curve:
        cost = A * cum^(-lambda)
    """

    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=1)
    df = df[["Year", cum_col, cost_col]].dropna()
    df = df[df[cum_col] > 0]
    df = df.rename(columns={
        cum_col: "installed_cap_gwh",
        cost_col: "bess_capex_kwh",
        "Year": "year"
    }).sort_values("year").reset_index(drop=True)

    X = np.log(df["installed_cap_gwh"].values.astype(float))
    y = np.log(df["bess_capex_kwh"].values.astype(float))

    b, a = np.polyfit(X, y, 1)
    lam = -b
    A = np.exp(a)
    learning_rate = 1 - 2**(-lam)

    def predict_cost(cap):
        cap = np.asarray(cap, float)
        return A * cap ** (-lam)

    print("\n================ Learning Curve Fit ================")
    print(f"λ (learning parameter):      {lam:.4f}")
    print(f"Learning rate per doubling:  {learning_rate*100:.1f}%")
    print(f"A:                           {A:.2f} $/kWh")
    print("Model: cost = A * cum^(-λ)")
    print("====================================================\n")

    return {
        "lambda": lam,
        "learning_rate": learning_rate,
        "A": A,
        "predict_cost": predict_cost,
        "df": df,
    }

if __name__ == "__main__":
    path = os.path.join(INPUT_PATH, "bess_installs_capex.xlsx")

    growth = fit_capacity_growth_curve(path)
    learn = fit_learning_curve(path)

    # historical cleaned
    df_hist = learn["df"]

    # future years
    start_year = int(df_hist["year"].iloc[-1])
    years = np.arange(start_year + 1, 2041)

    # step 1: project cumulative capacity from growth model
    proj_cap = growth["predict_capacity"](years)

    # step 2: project future cost from learning model
    proj_cost = learn["predict_cost"](proj_cap)

    df_future = pd.DataFrame({
        "year": years,
        "installed_cap_gwh": proj_cap,
        "bess_capex_kwh": proj_cost,
    })

    df_all = pd.concat([df_hist, df_future], ignore_index=True)

    out = os.path.join(INPUT_PATH, "bess_learning_curve.csv")
    df_all.to_csv(out, index=False)

    print("Saved:", out)
