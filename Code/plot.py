# --------------------------------------------------
# Plotting function (UPDATED)
# --------------------------------------------------
def plot_optimisation(optimal_bess_mwh, build_year, project_life, plot_data, best_result):
    """
    Plots capacity degradation (main plot) and discounted CAPEX outlays (sub-plot).
    """
    import matplotlib.pyplot as plt
    years = plot_data["years"]
    years_calendar = years + build_year

    # --- Data for Capacity Plot ---
    initial_cap = plot_data["initial_cap"]
    aug_cap = plot_data["aug_cap"]
    total_cap = initial_cap + aug_cap
    aug_year = best_result["augmentation_year"]
    aug_mwh = best_result["augmentation_capacity_mwh"]
    init_mwh = best_result["initial_capacity_mwh"]

    # --- Data for CAPEX Plot ---
    capex_disc_years = [build_year, aug_year]
    capex_disc_values = [
        best_result["initial_capex_disc"] / 1e6,  # Convert to M€
        best_result["augmentation_capex_disc"] / 1e6,  # Convert to M€
    ]
    capex_labels = ["Initial CAPEX", "Augmentation CAPEX"]

    # --- Create Dual Subplots ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 8),
        sharex=True,  # Share the X-axis (Year)
        gridspec_kw={'height_ratios': [3, 1]}  # Capacity plot is 3x taller than cost plot
    )
    plt.subplots_adjust(hspace=0.1)  # Reduce space between plots

    # =======================================================
    # Subplot 1: Capacity Degradation
    # =======================================================
    ax1.stackplot(
        years_calendar,
        initial_cap,
        aug_cap,
        labels=[
            f"Block 1 (Initial: {init_mwh:.1f} MWh)",
            f"Block 2 (Aug: {aug_mwh:.1f} MWh)"
        ],
        colors=['#1f77b4', '#ff7f0e']  # blue and orange
    )

    ax1.plot(
        years_calendar,
        [optimal_bess_mwh] * project_life,
        'k--',
        label=f"Minimum Required Capacity ({optimal_bess_mwh} MWh)",
        linewidth=2
    )

    ax1.axvline(
        aug_year,
        color='r',
        linestyle=':',
        linewidth=1,
        label=f"Augmentation Year ({aug_year})"
    )

    ax1.set_title(
        f'BESS Capacity and Discounted CAPEX Outlays for Optimal Strategy\n'
        f'(Best LCOE: €{best_result["levelised_cost_per_mwh"]:.2f}/MWh)',
        fontsize=14
    )
    ax1.set_ylabel('Available Capacity (MWh)')
    ax1.set_ylim(ymin=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', linestyle='--')

    # =======================================================
    # Subplot 2: Discounted CAPEX Outlay
    # =======================================================
    ax2.bar(
        capex_disc_years,
        capex_disc_values,
        color=['#1f77b4', '#ff7f0e'],
        width=0.8,
        label=capex_labels
    )

    # Add text labels on top of the bars
    for year, cost in zip(capex_disc_years, capex_disc_values):
        ax2.text(year, cost + cost * 0.05, f"€{cost:.1f}M", ha='center', va='bottom', fontsize=9)

    ax2.axvline(aug_year, color='r', linestyle=':', linewidth=1)  # Aug year marker

    ax2.set_xlabel('Project Year')
    ax2.set_ylabel('Discounted CAPEX (€M)')
    ax2.set_xlim(years_calendar[0], years_calendar[-1] + 1)  # Extend x-limit slightly
    ax2.set_ylim(ymin=0)
    ax2.ticklabel_format(style='plain', axis='y')  # Prevent scientific notation on Y-axis
    ax2.grid(True, axis='y', linestyle='--')

    # Create custom legend for the bar chart
    ax2.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc='#1f77b4'),
        plt.Rectangle((0, 0), 1, 1, fc='#ff7f0e')
    ], labels=capex_labels, loc='upper right')

    plt.tight_layout()
    plt.show()