import numpy as np
import pandas as pd
import os


from plot import plot_optimisation

CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")

# --------------------------------------------------
# Retention interpolation (NO CHANGE)
# --------------------------------------------------
def interpolate_retention(cycles_curve, retention_curve, cycles):
    """Linear interpolation for retention vs cycles."""
    cycles_curve = np.array(cycles_curve)
    retention_curve = np.array(retention_curve)
    cycles = np.array(cycles)
    return np.interp(
        cycles,
        cycles_curve,
        retention_curve,
        left=retention_curve[0],
        right=retention_curve[-1],
    )


# --------------------------------------------------
# Optimiser (BESS + one augmentation) (NO CHANGE)
# --------------------------------------------------
def optimise_bess_two_block_df(
        optimal_bess_mwh,
        cycles_per_annum,
        discount_rate,
        capex_df,  # <-- DataFrame with columns: year, cost
        cycles_curve,
        retention_curve,
        build_year,
        project_life,
        project_energy_gwh_per_annum,
):
    """
    Optimisation using a CAPEX dataframe. Returns the best result
    along with the necessary plotting data.
    """
    # Validate required columns
    required = ["year", "cost"]
    for col in required:
        if col not in capex_df.columns:
            raise ValueError(f"CAPEX DataFrame missing column: {col}")

    # Convert to lookup series
    capex_df = capex_df.set_index("year").sort_index()

    # Time indices for the project
    years = np.arange(project_life)  # 0,...,project_life-1
    discount = 1 / (1 + discount_rate) ** years

    project_energy_mwh = project_energy_gwh_per_annum * 1000
    pv_energy = (project_energy_mwh * discount).sum()

    # Compute initial retention curve
    cum_cycles = cycles_per_annum * (years + 0.5)
    retention_initial = interpolate_retention(cycles_curve, retention_curve, cum_cycles)

    best = None
    best_plot_data = None  # Store data needed for the best plot

    # Loop through possible augmentation years
    for aug_idx in range(1, project_life):

        aug_year = build_year + aug_idx

        # Check if CAPEX for augmentation year exists
        if aug_year not in capex_df.index:
            break

        # ----------------------------
        # Size initial block
        # ----------------------------
        min_ret_pre_aug = retention_initial[:aug_idx].min()
        initial_capacity_mwh = optimal_bess_mwh / min_ret_pre_aug

        # retention of the old block after augmentation
        cap_old_after_aug = initial_capacity_mwh * retention_initial[aug_idx:]

        # ----------------------------
        # Size augmentation block
        # ----------------------------
        years_after_aug = project_life - aug_idx
        cycles_new_block = cycles_per_annum * (np.arange(years_after_aug) + 0.5)
        retention_new = interpolate_retention(cycles_curve, retention_curve, cycles_new_block)

        required_each_year = (optimal_bess_mwh - cap_old_after_aug) / retention_new
        augmentation_capacity_mwh = max(0, required_each_year.max())

        # ----------------------------
        # Cost calculation
        # ----------------------------
        if build_year not in capex_df.index:
            raise ValueError(f"No CAPEX available for build year {build_year}")

        initial_capex_undisc = (initial_capacity_mwh * 1000) * capex_df.loc[build_year, "cost"]
        # Initial CAPEX is discounted at year 0 (i.e., not discounted)
        initial_capex_disc = initial_capex_undisc / ((1 + discount_rate) ** 0)

        aug_capex_undisc = (augmentation_capacity_mwh * 1000) * capex_df.loc[aug_year, "cost"]
        aug_capex_disc = aug_capex_undisc / ((1 + discount_rate) ** aug_idx)

        total_pv_cost = initial_capex_disc + aug_capex_disc
        levelised_cost = total_pv_cost / pv_energy

        result = {
            "augmentation_year_idx": aug_idx,  # Index needed for plotting
            "augmentation_year": aug_year,
            "initial_capacity_mwh": initial_capacity_mwh,
            "augmentation_capacity_mwh": augmentation_capacity_mwh,
            "initial_capex": initial_capex_undisc,  # Kept 'initial_capex' as undiscounted for printout clarity
            "initial_capex_disc": initial_capex_disc,  # NEW: Discounted value
            "augmentation_capex_undisc": aug_capex_undisc,
            "augmentation_capex_disc": aug_capex_disc,
            "levelised_cost_per_mwh": levelised_cost,
        }

        # Store plotting data for this scenario
        plot_data = {
            "years": years,
            "initial_cap": initial_capacity_mwh * retention_initial,
            "aug_cap": np.concatenate((
                np.zeros(aug_idx),
                augmentation_capacity_mwh * retention_new
            ))
        }

        if best is None or levelised_cost < best["levelised_cost_per_mwh"]:
            best = result
            best_plot_data = plot_data  # Update best plot data

    # Return the best result AND the plotting data
    return best, best_plot_data

# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    # --- Start of simplified CAPEX creation ---
    capex_data = pd.read_csv(os.path.join(INPUT_PATH, "bess_learning_curve.csv"))
    capex_df = capex_data.rename(columns={"bess_capex_kwh": "cost"})
    # --- End of simplified CAPEX creation ---

    cycles_curve = [0, 2000, 4000, 6000]
    retention_curve = [1.0, 0.96, 0.90, 0.80]
    optimal_bess_mwh_target = 10

    # The optimise function now returns the best result AND the plot data
    best_result, best_plot_data = optimise_bess_two_block_df(
        optimal_bess_mwh=optimal_bess_mwh_target,
        cycles_per_annum=300,
        discount_rate=0.08,
        capex_df=capex_df,
        cycles_curve=cycles_curve,
        retention_curve=retention_curve,
        build_year=2023,
        project_life=25,
        project_energy_gwh_per_annum=8.76,
    )

    print("\n" + "=" * 60)
    print("          🔋 BESS SIZING & AUGMENTATION RESULT")
    print("=" * 60)
    print(f"Initial capacity:            {best_result['initial_capacity_mwh']:.2f} MWh")
    print(f"Augmentation capacity:       {best_result['augmentation_capacity_mwh']:.2f} MWh")
    print(f"Augmentation year:           {best_result['augmentation_year']}")
    print("-" * 60)
    print(f"Initial CAPEX:               ${best_result['initial_capex']:,.0f} (Undiscounted)")
    print(f"Augmentation CAPEX (undisc): ${best_result['augmentation_capex_undisc']:,.0f}")
    print(f"Augmentation CAPEX (disc):   ${best_result['augmentation_capex_disc']:,.0f}")
    print("-" * 60)
    print(f"Levelised cost per MWh:      ${best_result['levelised_cost_per_mwh']:.2f}/MWh")
    print("=" * 60 + "\n")

    # Call the new plotting function
    plot_optimisation(
        optimal_bess_mwh=optimal_bess_mwh_target,
        build_year=2023,
        project_life=25,
        plot_data=best_plot_data,
        best_result=best_result
    )