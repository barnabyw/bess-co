import time
import pyomo.environ as pyo
import numpy as np
import csv

from assumptions import *
from profile import generate_hourly_solar_profile
from lcoe.lcoe import lcoe

#===Model Setup===
# -----------------------------

penalty_weight = 1e-3

def optimise_bess(solar_profile, capex_df, year):

    solar_cost_per_mw = capex_df.loc[capex_df["year"] == year, "solar_cost_per_mw"].values[0]
    bess_energy_cost_per_mwh = capex_df.loc[capex_df["year"] == year, "bess_energy_cost_per_mwh"].values[0]

    periods = len(solar_profile)
    demand = np.full(periods, load)
    T = range(periods)

    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=T)

    # Decision Variables
    model.solar_capacity = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_energy = pyo.Var(within=pyo.NonNegativeReals)

    model.bess_flow = pyo.Var(model.T, within=pyo.Reals)
    model.charge = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.energy_served_t = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # Constraints
    def soc_balance_rule(m, t):
        if t == 0:
            return m.soc[t] == m.bess_energy * start_soc
        # Negative flow = charging (adds to SOC), Positive flow = discharging (removes from SOC)
        return m.soc[t] == m.soc[t-1] - m.bess_flow[t] / efficiency
    model.soc_balance = pyo.Constraint(model.T, rule=soc_balance_rule)

    def energy_served_t_rule(m, t):
        return m.energy_served_t[t] == m.solar_capacity * solar_profile[t] + m.bess_flow[t]
    model.energy_served_t_constraint = pyo.Constraint(model.T, rule=energy_served_t_rule)

    model.bess_limits = pyo.ConstraintList()
    for t in T:
        model.bess_limits.add(model.soc[t] <= model.bess_energy)
        # Power limits
        model.bess_limits.add(model.bess_flow[t] <= model.solar_capacity)  # Max discharge
        model.bess_limits.add(model.bess_flow[t] >= -model.solar_capacity)  # Max charge
        # Can't discharge more than what's in battery
        model.bess_limits.add(model.bess_flow[t] <= model.soc[t] * efficiency)
        # Can't charge more than available solar
        model.bess_limits.add(-model.bess_flow[t] <= model.solar_capacity * solar_profile[t])
        # Penalty for simultaneous charge/discharge (if needed)
        # This might need adjustment based on your penalty logic

    model.energy_served_total = pyo.Constraint(expr=sum(model.energy_served_t[t] for t in T) >= target * sum(demand))

    model.energy_served_limit = pyo.ConstraintList()
    for t in T:
        model.energy_served_limit.add(model.energy_served_t[t] <= demand[t])

    # Objective Function
    model.cost = pyo.Objective(
        expr=model.solar_capacity * solar_cost_per_mw +
             model.bess_energy * bess_energy_cost_per_mwh,
        sense=pyo.minimize
    )

    # Solve
    print("Optimising...")
    start_time = time.time()
    from pyomo.opt import SolverFactory
    solver = SolverFactory('cbc')
    solver.solve(model)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Optimisation completed in {round(elapsed_time, 1)} seconds")

    print("Optimal Solar Capacity (MW):", pyo.value(model.solar_capacity))
    print("Optimal BESS Energy (MWh):", pyo.value(model.bess_energy))
    print("Cost:", round(pyo.value(model.cost),0))

    with open(r'C:\Users\barna\OneDrive\Documents\Solar_BESS results\optimization_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Hour', 'Solar (MW)', 'BESS flow (MW)', 'SOC (MWh)', 'Energy served (MW)'])
        for t in T:
            writer.writerow([
                t,
                solar_profile[t] * pyo.value(model.solar_capacity),
                pyo.value(model.bess_flow[t]),
                pyo.value(model.soc[t]),
                pyo.value(model.energy_served_t[t])
            ])

        levcost = 1000 * lcoe(load * 8760 * target,pyo.value(model.cost),0,0.08,20)
        print("lcoe:", round(levcost,1))

    return pyo.value(model.cost), pyo.value(model.solar_capacity), pyo.value(model.bess_energy), levcost

def optimise_availability(solar_profile, solar_capacity, bess_energy, load,
                          efficiency=efficiency, start_soc=start_soc):
    """
    Dispatch optimiser for fixed solar + BESS capacities.
    Maximises availability factor (fraction of demand served).

    Args:
        solar_profile (array): per-unit solar output [0–1] per timestep
        solar_capacity (float): installed solar capacity [MW]
        bess_energy (float): installed BESS energy capacity [MWh]
        load (float or array): demand per timestep [MW], scalar or array
        efficiency (float): round-trip efficiency (charge/discharge)
        start_soc (float): initial SoC as fraction of bess_energy [0–1]

    Returns:
        availability (float): fraction of demand met
        results (dict): dispatch time series
    """
    periods = len(solar_profile)
    T = range(periods)

    # Make demand vector
    if np.isscalar(load):
        demand = np.full(periods, load)
    else:
        demand = np.array(load)

    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=T)

    # Decision variables
    model.bess_flow = pyo.Var(model.T, within=pyo.Reals)  # Positive = discharge, Negative = charge
    model.soc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.energy_served_t = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # --- Constraints ---

    # SoC balance
    def soc_balance_rule(m, t):
        if t == 0:
            return m.soc[t] == bess_energy * start_soc
        # Negative flow = charging (adds to SOC), Positive flow = discharging (removes from SOC)
        return m.soc[t] == m.soc[t-1] - m.bess_flow[t] / efficiency
    model.soc_balance = pyo.Constraint(model.T, rule=soc_balance_rule)

    # Energy served
    def energy_served_t_rule(m, t):
        return m.energy_served_t[t] <= solar_capacity * solar_profile[t] + m.bess_flow[t]
    model.energy_served_t_constraint = pyo.Constraint(model.T, rule=energy_served_t_rule)

    # Storage & power limits
    model.limits = pyo.ConstraintList()
    for t in T:
        model.limits.add(model.soc[t] <= bess_energy)                                    # storage capacity
        model.limits.add(-model.bess_flow[t] <= solar_capacity * solar_profile[t])      # can only charge from available solar
        model.limits.add(model.bess_flow[t] <= solar_capacity)                          # inverter limit (max discharge)
        model.limits.add(model.bess_flow[t] >= -solar_capacity)                         # inverter limit (max charge)
        model.limits.add(model.bess_flow[t] <= model.soc[t] * efficiency)              # can't discharge more than available in battery
        model.limits.add(model.energy_served_t[t] <= demand[t])                        # can't serve more than demand

    # Objective: maximise total energy served
    model.obj = pyo.Objective(expr=sum(model.energy_served_t[t] for t in T),
                              sense=pyo.maximize)

    # --- Solve ---
    solver = pyo.SolverFactory("cbc")
    result = solver.solve(model, tee=False)

    if (result.solver.termination_condition == pyo.TerminationCondition.infeasible):
        print("⚠️ Infeasible model — returning availability = 0")
        return 0.0, {}

    # --- Results ---
    total_energy_served = sum(pyo.value(model.energy_served_t[t]) for t in T)
    total_demand = sum(demand)
    availability = total_energy_served / total_demand if total_demand > 0 else 0

    results = {
        "solar": [solar_profile[t] * solar_capacity for t in T],
        "bess_flow": [pyo.value(model.bess_flow[t]) for t in T],  # Single flow variable
        "soc": [pyo.value(model.soc[t]) for t in T],
        "energy_served": [pyo.value(model.energy_served_t[t]) for t in T]
    }

    return availability, results

if __name__ == "__main__":
    latitude = 19.4326
    longitude = 99.1332
    solar_profile = generate_hourly_solar_profile(latitude, longitude, solar_year=2023)
    print("got solar profile")
    cost, solar_capacity, bess_energy, levcost = optimise_bess(solar_profile, capex_learning_df, 2020)
    print(f"solar cap is {solar_capacity}, bess is {bess_energy}")
    availability, results = optimise_availability(solar_profile, solar_capacity, bess_energy, load=load)
    print(f"availability is {availability}")
    # Setting up environment
    """
    latitude = 19.4326
    longitude = 99.1332
    print("getting solar profile...")
    yearly_profile = generate_hourly_solar_profile(latitude, longitude, solar_year=2023)
    print("got solar profile")
    # demand_profile = np.full(len(yearly_profile), 100)  # Demand profile in MW

    # Run optimization
    optimise_bess(yearly_profile, capex_learning_df, 2020)
"""