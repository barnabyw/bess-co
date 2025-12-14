import numpy as np
import pandas as pd
import os
from plot import plot_optimisation
from bess_capex_cache import get_bess_capex_series

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
def optimise_augmentation(
    optimal_bess_mwh,
    cycles_per_annum,
    discount_rate,
    capex_opex_df,               # NEW: pass the real table
    build_year,
    project_life,
    project_energy_gwh_per_annum,
    country,
    scenario=None,
):
    cycles_curve = [0, 2000, 4000, 6000]
    retention_curve = [1.0, 0.96, 0.90, 0.80]

    # 1) Load CAPEX series (cached)
    capex_series = get_bess_capex_series(capex_opex_df, country=country, scenario=scenario)

    # --- EARLY EXIT: no BESS energy, or no cycling ---
    if optimal_bess_mwh <= 0 or cycles_per_annum <= 0:
        return {
            "augmentation_year_idx": None,
            "augmentation_year": None,
            "initial_capacity_mwh": optimal_bess_mwh,
            "augmentation_capacity_mwh": 0.0,
            "initial_capex_undisc": 0.0,
            "initial_capex_disc": 0.0,
            "augmentation_capex_undisc": 0.0,
            "augmentation_capex_disc": 0.0,
            "levelised_cost_per_mwh": 0.0,
        }, {
            "years": np.arange(project_life),
            "initial_cap": np.ones(project_life) * optimal_bess_mwh,
            "aug_cap": np.zeros(project_life)
        }

    # must contain build year entries
    if build_year not in capex_series.index:
        raise ValueError(f"No BESS capex_e available for build year={build_year}, scenario={scenario}")

    # work with a simple dict-like view
    def capex_undiscounted(mwh, year):
        if year not in capex_series.index:
            # stop augmentation if past available data
            return None
        return mwh * 1000 * capex_series.at[year]

    years = np.arange(project_life)
    discount_factors = (1 + discount_rate) ** (-years)
    pv_energy = project_energy_gwh_per_annum * 1000 * discount_factors.sum()

    # Precompute retention for initial block
    cum_cycles = cycles_per_annum * (years + 0.5)
    retention_initial = interpolate_retention(cycles_curve, retention_curve, cum_cycles)

    def capex_discounted(undisc_cost, year_idx):
        return undisc_cost / ((1 + discount_rate) ** year_idx)

    best = None
    best_plot_data = None

    for aug_idx in range(1, project_life):

        aug_year = build_year + aug_idx
        if aug_year not in capex_series.index:
            break

        # ----------------------------
        # Size initial block
        # ----------------------------
        min_ret_pre_aug = retention_initial[:aug_idx].min()
        initial_capacity_mwh = optimal_bess_mwh / min_ret_pre_aug
        cap_old_after_aug = initial_capacity_mwh * retention_initial[aug_idx:]

        # ----------------------------
        # Size augmentation block
        # ----------------------------
        years_after = project_life - aug_idx
        cycles_new = cycles_per_annum * (np.arange(years_after) + 0.5)
        retention_new = interpolate_retention(cycles_curve, retention_curve, cycles_new)

        augmentation_capacity_mwh = max(
            0, ((optimal_bess_mwh - cap_old_after_aug) / retention_new).max()
        )

        # ----------------------------
        # Costs (now storing undisc + disc)
        # ----------------------------
        initial_capex_undisc = capex_undiscounted(initial_capacity_mwh, build_year)
        initial_capex_disc = capex_discounted(initial_capex_undisc, 0)

        augmentation_capex_undisc = capex_undiscounted(augmentation_capacity_mwh, aug_year)
        augmentation_capex_disc = capex_discounted(augmentation_capex_undisc, aug_idx)

        levelised_cost = (initial_capex_disc + augmentation_capex_disc) / pv_energy

        result = {
            "augmentation_year_idx": aug_idx,
            "augmentation_year": aug_year,

            "initial_capacity_mwh": initial_capacity_mwh,
            "augmentation_capacity_mwh": augmentation_capacity_mwh,

            "initial_capex_undisc": initial_capex_undisc,
            "initial_capex_disc": initial_capex_disc,

            "augmentation_capex_undisc": augmentation_capex_undisc,
            "augmentation_capex_disc": augmentation_capex_disc,

            "levelised_cost_per_mwh": levelised_cost,
        }

        plot_data = {
            "years": years,
            "initial_cap": initial_capacity_mwh * retention_initial,
            "aug_cap": np.r_[np.zeros(aug_idx), augmentation_capacity_mwh * retention_new],
        }

        if best is None or levelised_cost < best["levelised_cost_per_mwh"]:
            best = result
            best_plot_data = plot_data

    return best, best_plot_data

if __name__ == "__main__":
    import pandas as pd
    import os

    CWD = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(CWD, "..", "inputs")

    # Load your real converted table (correct format for cache)
    capex_opex_df = pd.read_excel(os.path.join(INPUT_PATH, "capex_opex_converted.xlsx"))

    # === TEST PARAMETERS ===
    test_bess_mwh = 10
    test_cycles   = 300
    test_discount = 0.08
    test_year     = 2024
    test_life     = 25
    test_energy   = 8.76   # GWh per annum
    test_scenario = None   # or "Low", "High", etc.

    print("Loading BESS capex series...")
    series = get_bess_capex_series(capex_opex_df, test_scenario)
    print(series)

    print("\nRunning augmentation optimiser...\n")
    best_result, best_plot_data = optimise_augmentation(
        optimal_bess_mwh=test_bess_mwh,
        cycles_per_annum=test_cycles,
        discount_rate=test_discount,
        capex_opex_df=capex_opex_df,
        build_year=test_year,
        project_life=test_life,
        project_energy_gwh_per_annum=test_energy,
        scenario=test_scenario,
    )

    print("=== BEST AUGMENTATION RESULT ===")
    print(best_result)

    # Optional plotting
    try:
        from plot import plot_optimisation
        plot_optimisation(
            optimal_bess_mwh=test_bess_mwh,
            build_year=test_year,
            project_life=test_life,
            plot_data=best_plot_data,
            best_result=best_result,
        )
    except ImportError:
        print("plot_optimisation() not available – skipping plot.")
