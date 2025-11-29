import numpy as np
import pandas as pd
import os
from plot import plot_optimisation

CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")

# =========================================================
# Retention interpolation
# =========================================================
def interpolate_retention(cycles_curve, retention_curve, cycles):
    return np.interp(
        cycles,
        np.array(cycles_curve),
        np.array(retention_curve),
        left=retention_curve[0],
        right=retention_curve[-1],
    )

# =========================================================
# BESS optimiser (one augmentation block)
# =========================================================
def optimise_bess_two_block_df(
    optimal_bess_mwh,
    cycles_per_annum,
    discount_rate,
    capex_df,
    cycles_curve,
    retention_curve,
    build_year,
    project_life,
    project_energy_gwh_per_annum,
):
    # ---- Validate input ----
    if not {"year", "cost"} <= set(capex_df.columns):
        raise ValueError("CAPEX DataFrame must contain: year, cost")

    capex_df = capex_df.set_index("year").sort_index()

    years = np.arange(project_life)
    discount_factors = (1 + discount_rate) ** (-years)

    pv_energy = project_energy_gwh_per_annum * 1000 * discount_factors.sum()

    # Precompute retention for the *original* block
    cum_cycles = cycles_per_annum * (years + 0.5)
    retention_initial = interpolate_retention(cycles_curve, retention_curve, cum_cycles)

    # Helper for CAPEX
    def discounted_capex(mwh, year, t):
        cost = capex_df.at[year, "cost"]
        return mwh * 1000 * cost / ((1 + discount_rate) ** t)

    best = None
    best_plot_data = None

    for aug_idx in range(1, project_life):

        aug_year = build_year + aug_idx
        if aug_year not in capex_df.index:
            break  # Beyond available CAPEX years

        # ----------------------------
        # Initial block
        # ----------------------------
        min_ret_pre_aug = retention_initial[:aug_idx].min()
        initial_capacity_mwh = optimal_bess_mwh / min_ret_pre_aug
        cap_old_after_aug = initial_capacity_mwh * retention_initial[aug_idx:]

        # ----------------------------
        # Augmentation block
        # ----------------------------
        years_after = project_life - aug_idx
        cycles_new = cycles_per_annum * (np.arange(years_after) + 0.5)
        retention_new = interpolate_retention(cycles_curve, retention_curve, cycles_new)

        augmentation_capacity_mwh = max(
            0, ((optimal_bess_mwh - cap_old_after_aug) / retention_new).max()
        )

        # ----------------------------
        # Costs
        # ----------------------------
        initial_capex_disc = discounted_capex(initial_capacity_mwh, build_year, 0)
        aug_capex_disc = discounted_capex(augmentation_capacity_mwh, aug_year, aug_idx)

        levelised_cost = (initial_capex_disc + aug_capex_disc) / pv_energy

        result = {
            "augmentation_year_idx": aug_idx,
            "augmentation_year": aug_year,
            "initial_capacity_mwh": initial_capacity_mwh,
            "augmentation_capacity_mwh": augmentation_capacity_mwh,
            "initial_capex_disc": initial_capex_disc,
            "augmentation_capex_disc": aug_capex_disc,
            "levelised_cost_per_mwh": levelised_cost,
        }

        plot_data = {
            "years": years,
            "initial_cap": initial_capacity_mwh * retention_initial,
            "aug_cap": np.r_[np.zeros(aug_idx), augmentation_capacity_mwh * retention_new],
        }

        if best is None or levelised_cost < best["levelised_cost_per_mwh"]:
            best, best_plot_data = result, plot_data

    return best, best_plot_data

# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    capex_data = pd.read_csv(os.path.join(INPUT_PATH, "bess_learning_curve.csv"))
    capex_df = capex_data.rename(columns={"bess_capex_kwh": "cost"})

    cycles_curve = [0, 2000, 4000, 6000]
    retention_curve = [1.0, 0.96, 0.90, 0.80]

    best_result, best_plot_data = optimise_bess_two_block_df(
        optimal_bess_mwh=10,
        cycles_per_annum=300,
        discount_rate=0.08,
        capex_df=capex_df,
        cycles_curve=cycles_curve,
        retention_curve=retention_curve,
        build_year=2023,
        project_life=25,
        project_energy_gwh_per_annum=8.76,
    )

    # Print summary
    print(best_result)

    plot_optimisation(
        optimal_bess_mwh=10,
        build_year=2023,
        project_life=25,
        plot_data=best_plot_data,
        best_result=best_result,
    )
