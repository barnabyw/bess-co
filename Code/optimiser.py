import time

import pyomo.environ as pyo
import numpy as np
import csv

from assumptions import *
from profile import generate_hourly_solar_profile

#===Model Setup===
# -----------------------------

solar_profile = np.array([
    0.0001255, 0.0005609, 0.0021880, 0.0074480, 0.0221268, 0.0573706,
    0.1298224, 0.2563892, 0.4419161, 0.6647677, 0.8727502, 1.0000000,
    1.0000000, 0.8727502, 0.6647677, 0.4419161, 0.2563892, 0.1298224,
    0.0573706, 0.0221268, 0.0074480, 0.0021880, 0.0005609, 0.0001255,

    0.0001255, 0.0005609, 0.0021880, 0.0074480, 0.0221268, 0.0573706,
    0.1298224, 0.2563892, 0.4419161, 0.6647677, 0.8727502, 1.0000000,
    1.0000000, 0.8727502, 0.6647677, 0.4419161, 0.2563892, 0.1298224,
    0.0573706, 0.0221268, 0.0074480, 0.0021880, 0.0005609, 0.0001255,

    0.0001255, 0.0005609, 0.0021880, 0.0074480, 0.0221268, 0.0573706,
    0.1298224, 0.2563892, 0.4419161, 0.6647677, 0.8727502, 1.0000000,
    1.0000000, 0.8727502, 0.6647677, 0.4419161, 0.2563892, 0.1298224,
    0.0573706, 0.0221268, 0.0074480, 0.0021880, 0.0005609, 0.0001255
])

def optimise_bess(solar_profile, capex_df, year):

    solar_cost_per_mw = capex_df.loc[capex_df["year"] == year, "solar_cost_per_mw"].values[0]
    bess_power_cost_per_mw = capex_df.loc[capex_df["year"] == year, "bess_power_cost_per_mw"].values[0]
    bess_energy_cost_per_mwh = capex_df.loc[capex_df["year"] == year, "bess_energy_cost_per_mwh"].values[0]

    periods = len(solar_profile)
    demand = np.full(periods, load)
    T = range(periods)

    #===Model Definition===
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=T)

    #===Decision Variables===
    model.solar_capacity = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_power = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_energy = pyo.Var(within=pyo.NonNegativeReals)

    model.charge = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.energy_served_t = pyo.Var(model.T, within=pyo.NonNegativeReals)

    model.is_charging = pyo.Var(model.T, within=pyo.Binary)
    model.is_discharging = pyo.Var(model.T, within=pyo.Binary)

    # -----------------------------
    # Constraints
    # -----------------------------

    # SOC balance
    def soc_balance_rule(m, t):
        if t == 0:
            return m.soc[t] == m.bess_energy * start_soc # or fix to 0 if preferred
        return m.soc[t] == m.soc[t-1] + m.charge[t] * efficiency - m.discharge[t] / efficiency
    model.soc_balance = pyo.Constraint(model.T, rule=soc_balance_rule)

    def energy_served_t_rule(m, t):
        return m.energy_served_t[t] == m.solar_capacity * solar_profile[t] + m.discharge[t] - m.charge[t]
    model.energy_served_t_constraint = pyo.Constraint(model.T, rule=energy_served_t_rule)

    # BESS operational limits and mutual exclusivity
    def bess_constraints(m, t):
        return [
            m.charge[t] <= M * m.is_charging[t],
            m.discharge[t] <= M * m.is_discharging[t],
            m.is_charging[t] + m.is_discharging[t] <= 1,
            m.soc[t] <= m.bess_energy,
            m.charge[t] <= m.bess_power,
            m.discharge[t] <= m.bess_power
        ]
    model.bess_limits = pyo.ConstraintList()
    for t in T:
        for c in bess_constraints(model, t):
            model.bess_limits.add(c)

    # Energy served must meet 95% of demand
    model.energy_served_total = pyo.Constraint(expr=sum(model.energy_served_t[t] for t in T) >= 0.95 * sum(demand))

    # Per-period limits
    model.energy_served_limit = pyo.ConstraintList()
    for t in T:
        model.energy_served_limit.add(model.energy_served_t[t] <= demand[t])
        model.energy_served_limit.add(model.discharge[t] <= model.soc[t])
        model.energy_served_limit.add(model.charge[t] <= model.solar_capacity * solar_profile[t])

    # -----------------------------
    # Objective Function
    # -----------------------------
    model.cost = pyo.Objective(
        expr=model.solar_capacity * solar_cost_per_mw +
             model.bess_power * bess_power_cost_per_mw +
             model.bess_energy * bess_energy_cost_per_mwh,
        sense=pyo.minimize
    )

    # -----------------------------
    # Solve
    # -----------------------------
    print("Optimising...")
    start_time = time.time()
    solver = pyo.SolverFactory('cbc')
    solver.solve(model)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"the optimisation took {round(elapsed_time,1)} secs")
    # -----------------------------
    # Output
    # -----------------------------
    print("Optimal Solar Capacity (MW):", pyo.value(model.solar_capacity))
    print("Optimal BESS Power (MW):", pyo.value(model.bess_power))
    print("Optimal BESS Energy (MWh):", pyo.value(model.bess_energy))
    print("Cost:", round(pyo.value(model.cost), 0))

    with open(r'C:\Users\barnaby.winser\Documents\solar bes\optimization_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Hour', 'Solar', 'Charge (MW)', 'Discharge (MW)', 'SOC (MWh)', 'Energy Served (MW)'])
        for t in T:
            writer.writerow([
                t,
                solar_profile[t] * pyo.value(model.solar_capacity),
                pyo.value(model.charge[t]),
                pyo.value(model.discharge[t]),
                pyo.value(model.soc[t]),
                pyo.value(model.energy_served_t[t])
            ])

    return pyo.value(model.cost), pyo.value(model.solar_capacity), pyo.value(model.bess_power), pyo.value(model.bess_energy)

if __name__ == "__main__":
    # Setting up environment
    latitude = 36.1716
    longitude = 115.1391
    print("getting solar profile...")
    yearly_profile = generate_hourly_solar_profile(latitude, longitude, year=2023)
    print("got solar profile")
    # demand_profile = np.full(len(yearly_profile), 100)  # Demand profile in MW

    # Run optimization
    optimise_bess(yearly_profile, capex_learning_df, year=2020)