import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# --------------------------------------------------
# Retention interpolation
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
# Optimiser (BESS + one augmentation)
# CAPEX sourced directly from DataFrame
# --------------------------------------------------
def optimise_bess_two_block_df(
    optimal_bess_mwh,
    cycles_per_annum,
    discount_rate,
    capex_df,                      # <-- DataFrame with columns: year, cost
    cycles_curve,
    retention_curve,
    build_year,
    project_life,
    project_energy_gwh_per_annum,
):
    """
    optimisation using a CAPEX dataframe:
        year | installed_cap_gwh | bess_capex_kwh

    Only column used here is:
        year
        cost (renamed from bess_capex_kwh)
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

    # Loop through possible augmentation years
    for aug_idx in range(1, project_life):

        aug_year = build_year + aug_idx

        # Check if CAPEX for augmentation year exists
        if aug_year not in capex_df.index:
            # user said "OK for now 2025 last available year"
            # so we simply stop searching once year runs out
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
        # initial cost at build year
        if build_year not in capex_df.index:
            raise ValueError(f"No CAPEX available for build year {build_year}")

        initial_capex = (initial_capacity_mwh * 1000) * capex_df.loc[build_year, "cost"]

        # augmentation cost at aug_year
        aug_capex_undisc = (augmentation_capacity_mwh * 1000) * capex_df.loc[aug_year, "cost"]
        aug_capex_disc = aug_capex_undisc / ((1 + discount_rate) ** aug_idx)

        total_pv_cost = initial_capex + aug_capex_disc
        levelised_cost = total_pv_cost / pv_energy

        result = {
            "augmentation_year": aug_year,
            "initial_capacity_mwh": initial_capacity_mwh,
            "augmentation_capacity_mwh": augmentation_capacity_mwh,
            "initial_capex": initial_capex,
            "augmentation_capex_undisc": aug_capex_undisc,
            "augmentation_capex_disc": aug_capex_disc,
            "levelised_cost_per_mwh": levelised_cost,
        }

        if best is None or levelised_cost < best["levelised_cost_per_mwh"]:
            best = result

    return best


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":

    # Load CAPEX learning curve CSV
    CWD = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(CWD, "..", "inputs")
    CAPEX_CSV = os.path.join(INPUT_PATH, "bess_learning_curve.csv")

    capex_df = pd.read_csv(CAPEX_CSV)

    # Rename to standard column names the optimiser expects
    capex_df = capex_df.rename(columns={
        "bess_capex_kwh": "cost",
        "installed_cap_gwh": "cum"
    })

    cycles_curve = [0, 2000, 4000, 6000]
    retention_curve = [1.0, 0.96, 0.90, 0.80]

    result = optimise_bess_two_block_df(
        optimal_bess_mwh=10,
        cycles_per_annum=300,
        discount_rate=0.08,
        capex_df=capex_df,
        cycles_curve=cycles_curve,
        retention_curve=retention_curve,
        build_year=2023,               # last available CAPEX year is 2025 → OK
        project_life=25,
        project_energy_gwh_per_annum=8.76,
    )

    print("\n" + "=" * 60)
    print("          🔋 BESS SIZING & AUGMENTATION RESULT")
    print("=" * 60)
    print(f"Initial capacity:            {result['initial_capacity_mwh']:.2f} MWh")
    print(f"Augmentation capacity:       {result['augmentation_capacity_mwh']:.2f} MWh")
    print(f"Augmentation year:           {result['augmentation_year']}")
    print("-" * 60)
    print(f"Initial CAPEX:               €{result['initial_capex']:,.0f}")
    print(f"Augmentation CAPEX (undisc): €{result['augmentation_capex_undisc']:,.0f}")
    print(f"Augmentation CAPEX (disc):   €{result['augmentation_capex_disc']:,.0f}")
    print("-" * 60)
    print(f"Levelised cost per MWh:      €{result['levelised_cost_per_mwh']:.2f}/MWh")
    print("=" * 60 + "\n")
