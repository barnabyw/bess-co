import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =======================================================
# CONSTANTS & PATHS
# =======================================================
CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")

Z_5_95 = 1.6448536269514722
N_MC = 5000  # Monte Carlo iterations


# =======================================================
# LOGISTIC MODEL (YEAR → CAPACITY) with uncertainty
# =======================================================
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def fit_logistic_capacity(path, cum_col="gwh capacity", sheet_name="battery_kwh", K=100000):
    """Fit logistic with fixed K and return uncertainty for (r, t0)."""

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

    def logi(t, r, t0):
        return logistic(t, K, r, t0)

    params, cov = curve_fit(
        logi,
        years,
        cap,
        p0=[r_guess, t0_guess],
        maxfev=20000
    )

    r, t0 = params
    se_r = np.sqrt(cov[0, 0])
    se_t0 = np.sqrt(cov[1, 1])

    r_low = r - Z_5_95 * se_r
    r_high = r + Z_5_95 * se_r

    t0_low = t0 - Z_5_95 * se_t0
    t0_high = t0 + Z_5_95 * se_t0

    def predict_capacity(t, r_val=r, t0_val=t0):
        t = np.asarray(t, float)
        return logistic(t, K, r_val, t0_val)

    print("\n========== Logistic Capacity Fit ==========")
    print(f"K fixed at:            {K/1000:.1f} TWh")
    print(f"Growth rate r:         {r:.4f}")
    print(f"r (5–95% CI):          {r_low:.4f} → {r_high:.4f}")
    print(f"Inflection year t0:    {t0:.2f}")
    print(f"t0 (5–95% CI):         {t0_low:.2f} → {t0_high:.2f}")
    print("===========================================\n")

    return {
        "K": K,
        "r": r,
        "t0": t0,
        "r_se": se_r,
        "t0_se": se_t0,
        "r_ci": (r_low, r_high),
        "t0_ci": (t0_low, t0_high),
        "cov": cov,
        "predict_capacity": predict_capacity,
        "df": df
    }


# =======================================================
# LEARNING CURVE FIT (CAPACITY → COST) + UNCERTAINTY
# =======================================================
def fit_learning_curve_with_uncertainty(path, cum_col="gwh capacity", cost_col="$/kWh", sheet_name='battery_kwh'):

    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=1)
    df = df[["Year", cum_col, cost_col]].dropna()
    df = df[df[cum_col] > 0].rename(columns={
        cum_col: "installed_cap_gwh",
        cost_col: "bess_capex_kwh",
        "Year": "year"
    }).sort_values("year").reset_index(drop=True)

    X = np.log(df["installed_cap_gwh"].values)
    y = np.log(df["bess_capex_kwh"].values)

    # regression
    b, a = np.polyfit(X, y, 1)
    lam = -b
    A = np.exp(a)

    # variance & covariance
    y_fit = a + b * X
    residuals = y - y_fit
    s2 = np.sum(residuals**2) / (len(X) - 2)

    Xmat = np.vstack([np.ones(len(X)), X]).T
    cov_ab = s2 * np.linalg.inv(Xmat.T @ Xmat)

    print("\n================ Learning Curve Fit ================")
    print(f"Lambda (λ):                {lam:.4f}")
    print(f"Learning rate:             {(1 - 2**(-lam))*100:.2f}% per doubling")
    print("====================================================\n")

    return {
        "lambda": lam,
        "A": A,
        "cov_ab": cov_ab,
        "df": df
    }


# =======================================================
# MONTE CARLO (FULL: K + learning params + logistic r/t0)
# =======================================================
def monte_carlo_simulation(years, logistic_fit, learn_fit,
                           K_dist=(85_000, 112_500, 140_000), n_mc=2000):

    # extract logistic params
    r_c = logistic_fit["r"]
    t0_c = logistic_fit["t0"]
    r_se = logistic_fit["r_se"]
    t0_se = logistic_fit["t0_se"]
    K_mid = K_dist[1]

    # learning curve regression params
    A_c = learn_fit["A"]
    lam_c = learn_fit["lambda"]
    b_c = -lam_c
    a_c = np.log(A_c)
    cov_ab = learn_fit["cov_ab"]

    T = len(years)
    mc_cost = np.zeros((n_mc, T))

    for i in range(n_mc):

        # 1. Sample logistic parameters
        r_s = np.random.normal(r_c, r_se)
        t0_s = np.random.normal(t0_c, t0_se)

        # 2. Sample K (capacity uncertainty)
        K_s = np.random.triangular(*K_dist)

        # 3. Compute capacity path
        cap = logistic(years, K_s, r_s, t0_s)

        # 4. Sample learning parameters
        a_s, b_s = np.random.multivariate_normal([a_c, b_c], cov_ab)
        A_s = np.exp(a_s)
        lam_s = -b_s

        # 5. Compute cost path
        mc_cost[i, :] = A_s * cap ** (-lam_s)

    # percentiles
    return {
        "P10": np.percentile(mc_cost, 10, axis=0),
        "P50": np.percentile(mc_cost, 50, axis=0),
        "P90": np.percentile(mc_cost, 90, axis=0)
    }


# =======================================================
# MAIN SCRIPT
# =======================================================
if __name__ == "__main__":

    path = os.path.join(INPUT_PATH, "bess_installs_capex.xlsx")

    # 1. Fit logistic with uncertainty
    K_mid = 112_500
    logistic_fit = fit_logistic_capacity(path, K=K_mid)

    # 2. Fit learning curve
    learn_fit = fit_learning_curve_with_uncertainty(path)

    # Projection years
    last_hist_year = int(learn_fit["df"]["year"].iloc[-1])
    years = np.arange(max(2024, last_hist_year + 1), 2041)

    # 3. Run full MC simulation
    combined = monte_carlo_simulation(
        years,
        logistic_fit,
        learn_fit,
        K_dist=(85_000, 112_500, 140_000),
        n_mc=N_MC
    )

    # 4. Save output
    df_mc = pd.DataFrame({
        "year": years,
        "P10_cost": combined["P10"],
        "P50_cost": combined["P50"],
        "P90_cost": combined["P90"]
    })

    output_file = os.path.join(INPUT_PATH, "bess_mc_cost_projection.csv")
    df_mc.to_csv(output_file, index=False)
    print(f"\nMonte Carlo results saved to:\n  {output_file}\n")

    # 5. Plot
    plt.figure(figsize=(14, 8))
    plt.plot(years, combined["P50"], color="black", linewidth=2, label="Central (P50)")
    plt.fill_between(years, combined["P10"], combined["P90"],
                     color="orange", alpha=0.3, label="Full MC (P10–P90)")

    plt.title("Monte Carlo BESS Cost Projection (r, t0, K, λ, A uncertainty)")
    plt.xlabel("Year")
    plt.ylabel("Cost ($/kWh)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
