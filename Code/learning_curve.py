import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =======================================================
# Paths & constants
# =======================================================
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")

# 5–95% normal quantile (two-sided)
Z_5_95 = 1.6448536269514722


# =======================================================
# Logistic capacity model (YEAR → cumulative GWh)
# =======================================================
def logistic(t, K, r, t0):
    """Standard logistic function."""
    return K / (1 + np.exp(-r * (t - t0)))


def fit_logistic_capacity(path, cum_col="gwh capacity", sheet_name="battery_kwh", K=100000):
    """
    Fit logistic curve with fixed saturation K (in GWh):

        C(t) = K / (1 + exp(-r (t - t0)))

    Returns:
        {
            "K": K,
            "r": r,
            "t0": t0,
            "predict_capacity": function(year),
            "df": cleaned historical dataframe
        }
    """

    # Load + clean
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=1)
    df = df[["Year", cum_col]].dropna()
    df = df[df[cum_col] > 0].rename(columns={
        cum_col: "installed_cap_gwh",
        "Year": "year"
    }).sort_values("year")

    years = df["year"].values.astype(float)
    cap = df["installed_cap_gwh"].values.astype(float)

    # Initial guesses
    t0_guess = np.median(years)
    r_guess = 0.15

    def logistic_fixed(t, r, t0):
        return logistic(t, K, r, t0)

    params, _ = curve_fit(
        logistic_fixed,
        years,
        cap,
        p0=[r_guess, t0_guess],
        maxfev=20000
    )
    r, t0 = params

    def predict_capacity(t):
        t = np.asarray(t, float)
        return logistic(t, K, r, t0)

    print("\n========== Logistic Capacity Fit ==========")
    print(f"K fixed at:          {K/1000:.1f} TWh")
    print(f"Growth rate r:       {r:.4f}")
    print(f"Inflection year t0:  {t0:.1f}")
    print("===========================================\n")

    return {
        "K": K,
        "r": r,
        "t0": t0,
        "predict_capacity": predict_capacity,
        "df": df
    }


# =======================================================
# Learning curve model (capacity → cost) with uncertainty
# =======================================================
def fit_learning_curve_with_uncertainty(path, cum_col="gwh capacity", cost_col="$/kWh", sheet_name='battery_kwh'):
    """
    Fits a learning curve:

        cost = A * cap^(-lambda)

    via log-log regression:

        log(cost) = a + b * log(cap)
        lambda = -b, A = exp(a)

    Also computes a 5–95% confidence band on log(cost) predictions.

    Returns:
        {
          "lambda": lam,
          "lambda_ci": (lam_low, lam_high),
          "learning_rate": lr,
          "learning_rate_ci": (lr_low, lr_high),
          "A": A,
          "predict_cost": f(cap) -> central cost,
          "predict_cost_with_ci": f(cap) -> (central, low, high),
          "df": cleaned historical dataframe
        }
    """

    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=1)
    df = df[["Year", cum_col, cost_col]].dropna()
    df = df[df[cum_col] > 0].rename(columns={
        cum_col: "installed_cap_gwh",
        cost_col: "bess_capex_kwh",
        "Year": "year"
    }).sort_values("year").reset_index(drop=True)

    X = np.log(df["installed_cap_gwh"].astype(float).values)
    y = np.log(df["bess_capex_kwh"].astype(float).values)
    n = len(X)

    # Linear regression in log-log space
    b, a = np.polyfit(X, y, 1)
    lam = -b
    A = np.exp(a)
    lr = 1 - 2**(-lam)  # learning rate per doubling

    # Residuals and variance
    y_fit = a + b * X
    residuals = y - y_fit
    s2 = np.sum(residuals**2) / (n - 2)

    # Covariance matrix of [a, b]
    Xmat = np.vstack([np.ones(n), X]).T
    cov = s2 * np.linalg.inv(Xmat.T @ Xmat)
    se_a = np.sqrt(cov[0, 0])
    se_b = np.sqrt(cov[1, 1])

    # Lambda uncertainty
    se_lambda = se_b
    lam_low = lam - Z_5_95 * se_lambda
    lam_high = lam + Z_5_95 * se_lambda

    # Learning-rate uncertainty (derived from λ bounds)
    lr_low = 1 - 2**(-lam_high)   # slower learning → higher λ? careful: learning rate increases with λ
    lr_high = 1 - 2**(-lam_low)

    def predict_cost(cap):
        """Central cost forecast (no CI)."""
        cap = np.asarray(cap, float)
        Xnew = np.log(cap)
        yhat = a + b * Xnew
        return np.exp(yhat)

    def predict_cost_with_ci(cap):
        """
        Returns (central, low, high) where:
          - 'low'  = upper log-cost bound (high cost, 5% tail)
          - 'high' = lower log-cost bound (low cost, 95% tail)
        i.e. a 5–95% band in log-space, mapped back to cost-space.
        """
        cap = np.asarray(cap, float)
        Xnew = np.log(cap)
        yhat = a + b * Xnew

        # Var(yhat) = x' Σ x
        var_yhat = (
            se_a**2
            + Xnew**2 * se_b**2
            + 2 * Xnew * cov[0, 1]
        )
        se_yhat = np.sqrt(var_yhat)

        y_low = yhat + Z_5_95 * se_yhat   # upper log-cost (5% tail, high cost)
        y_high = yhat - Z_5_95 * se_yhat  # lower log-cost (95% tail, low cost)

        central = np.exp(yhat)
        low = np.exp(y_low)
        high = np.exp(y_high)
        return central, low, high

    print("\n================ Learning Curve Fit ================")
    print(f"Lambda (λ):                    {lam:.4f}")
    print(f"λ (5–95% CI):                  {lam_low:.4f} → {lam_high:.4f}")
    print()
    print(f"Central learning rate:         {lr*100:.2f}% per doubling")
    print(f"Learning rate (5% bound):      {lr_low*100:.2f}%")
    print(f"Learning rate (95% bound):     {lr_high*100:.2f}%")
    print("====================================================\n")

    return {
        "lambda": lam,
        "lambda_ci": (lam_low, lam_high),
        "learning_rate": lr,
        "learning_rate_ci": (lr_low, lr_high),
        "A": A,
        "predict_cost": predict_cost,
        "predict_cost_with_ci": predict_cost_with_ci,
        "df": df,
    }


# =======================================================
# Main workflow
# =======================================================
if __name__ == "__main__":

    # ----------------------------
    # Inputs
    # ----------------------------
    input_file = os.path.join(INPUT_PATH, "bess_installs_capex.xlsx")

    # Saturation scenarios 85–140 TWh (convert to GWh)
    K_low = 85_000      # slow adoption → low capacity → high cost
    K_mid = 112_500     # central case
    K_high = 140_000    # fast adoption → high capacity → low cost

    # ----------------------------
    # Fit logistic capacity models
    # ----------------------------
    fit_low = fit_logistic_capacity(input_file, K=K_low)
    fit_mid = fit_logistic_capacity(input_file, K=K_mid)
    fit_high = fit_logistic_capacity(input_file, K=K_high)

    # ----------------------------
    # Fit learning curve + uncertainty
    # ----------------------------
    learn = fit_learning_curve_with_uncertainty(input_file)

    # ----------------------------
    # Projection years
    # ----------------------------
    # Start from last historical year + 1 up to 2040
    last_hist_year = int(learn["df"]["year"].iloc[-1])
    start_year = max(last_hist_year + 1, 2024)
    end_year = 2040
    years = np.arange(start_year, end_year + 1)

    # ----------------------------
    # Capacity projections
    # ----------------------------
    cap_low = fit_low["predict_capacity"](years)   # slow adoption
    cap_mid = fit_mid["predict_capacity"](years)   # central adoption
    cap_high = fit_high["predict_capacity"](years) # fast adoption

    # ----------------------------
    # Cost projections – central learning curve (no LR uncertainty yet)
    # ----------------------------
    cost_central = learn["predict_cost"](cap_mid)

    # Capacity-driven cost uncertainty band (central λ)
    cost_cap_low = learn["predict_cost"](cap_low)    # low capacity → higher cost
    cost_cap_high = learn["predict_cost"](cap_high)  # high capacity → lower cost

    # ----------------------------
    # Learning-curve uncertainty band (conditional on central capacity path)
    # ----------------------------
    cost_central2, cost_lr_high_cost, cost_lr_low_cost = learn["predict_cost_with_ci"](cap_mid)
    # cost_lr_high_cost = high-cost bound (5% tail)
    # cost_lr_low_cost  = low-cost bound (95% tail)

    # Sanity check: central predictions should match
    # (not strictly necessary, but nice to be sure)
    # They may differ by tiny numerical noise.
    # print("Max diff central:", np.max(np.abs(cost_central - cost_central2)))

    # ----------------------------
    # Build dataframe with all pieces
    # ----------------------------
    df_proj = pd.DataFrame({
        "year": years,

        # Capacity scenarios
        "cap_low_gwh": cap_low,
        "cap_central_gwh": cap_mid,
        "cap_high_gwh": cap_high,

        # Central cost path (central capacity + central learning)
        "cost_central_kwh": cost_central,

        # Capacity-only uncertainty band
        # (difference in deployment, fixed learning curve)
        "cost_cap_high_cost_kwh": cost_cap_low,   # slow adoption → high cost
        "cost_cap_low_cost_kwh": cost_cap_high,   # fast adoption → low cost

        # Learning-curve uncertainty band
        # (regression uncertainty at central capacity path)
        "cost_lr_high_cost_kwh": cost_lr_high_cost,  # upper CI (5% tail, high cost)
        "cost_lr_low_cost_kwh": cost_lr_low_cost,    # lower CI (95% tail, low cost)
    })

    # ----------------------------
    # Print some of the projection table
    # ----------------------------
    print("\n========== Projection ({}–{}) ==========".format(start_year, end_year))
    print(df_proj.head())
    print("...\n")
    print(df_proj.tail())
    print("========================================\n")

    # ----------------------------
    # Save to CSV
    # ----------------------------
    output_file = os.path.join(INPUT_PATH, "bess_capacity_cost_projection_with_uncertainty.csv")
    df_proj.to_csv(output_file, index=False)
    print(f"Saved projection CSV to:\n  {output_file}\n")

    # ----------------------------
    # Plot: central line + capacity band + learning band
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Central line
    ax.plot(years, cost_central, label="Central cost (central K, central learning)", linewidth=2)

    # Capacity-only band (narrower band)
    cap_band_lower = np.minimum(cost_cap_low, cost_cap_high)
    cap_band_upper = np.maximum(cost_cap_low, cost_cap_high)
    ax.fill_between(
        years,
        cap_band_lower,
        cap_band_upper,
        alpha=0.25,
        label="Capacity uncertainty (K_low–K_high, fixed learning)"
    )

    # Learning-curve band (wider band, on central capacity path)
    lr_band_lower = np.minimum(cost_lr_high_cost, cost_lr_low_cost)
    lr_band_upper = np.maximum(cost_lr_high_cost, cost_lr_low_cost)
    ax.fill_between(
        years,
        lr_band_lower,
        lr_band_upper,
        alpha=0.15,
        label="Learning curve uncertainty (5–95%, central capacity)"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("BESS CAPEX ($/kWh)")
    ax.set_title("BESS Cost Projection with Capacity and Learning-Curve Uncertainty")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
