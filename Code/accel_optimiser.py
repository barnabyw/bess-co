import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import csv
import time

from assumptions import *
from profile import generate_hourly_solar_profile
from lcoe.lcoe import lcoe

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

penalty_weight = 1e-3

def optimise_bess(solar_profile):
    periods = len(solar_profile)
    demand = np.full(periods, load)
    T = range(periods)

    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=T)

    # Decision Variables
    model.solar_capacity = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_power = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_energy = pyo.Var(within=pyo.NonNegativeReals)

    model.charge = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.energy_served_t = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.penalty = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # Constraints
    def soc_balance_rule(m, t):
        if t == 0:
            return m.soc[t] == m.bess_energy * start_soc
        return m.soc[t] == m.soc[t-1] + m.charge[t] * efficiency - m.discharge[t] / efficiency
    model.soc_balance = pyo.Constraint(model.T, rule=soc_balance_rule)

    def energy_served_t_rule(m, t):
        return m.energy_served_t[t] == m.solar_capacity * solar_profile[t] + m.discharge[t] - m.charge[t]
    model.energy_served_t_constraint = pyo.Constraint(model.T, rule=energy_served_t_rule)

    model.bess_limits = pyo.ConstraintList()
    for t in T:
        model.bess_limits.add(model.soc[t] <= model.bess_energy)
        model.bess_limits.add(model.charge[t] <= model.bess_power)
        model.bess_limits.add(model.discharge[t] <= model.bess_power)
        model.bess_limits.add(model.discharge[t] <= model.soc[t])
        model.bess_limits.add(model.charge[t] <= model.solar_capacity * solar_profile[t])
        model.bess_limits.add(model.penalty[t] >= model.charge[t] + model.discharge[t] - model.bess_power)

    model.energy_served_total = pyo.Constraint(expr=sum(model.energy_served_t[t] for t in T) >= target * sum(demand))

    model.energy_served_limit = pyo.ConstraintList()
    for t in T:
        model.energy_served_limit.add(model.energy_served_t[t] <= demand[t])

    # Objective Function
    model.cost = pyo.Objective(
        expr=model.solar_capacity * solar_cost_per_mw +
             model.bess_power * bess_power_cost_per_mw +
             model.bess_energy * bess_energy_cost_per_mwh +
             penalty_weight * sum(model.penalty[t] for t in T),
        sense=pyo.minimize
    )

    # Solve
    print("Optimising...")
    start_time = time.time()
    solver = SolverFactory('cbc')
    solver.solve(model)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Optimisation completed in {round(elapsed_time, 1)} seconds")

    print("Optimal Solar Capacity (MW):", pyo.value(model.solar_capacity))
    print("Optimal BESS Power (MW):", pyo.value(model.bess_power))
    print("Optimal BESS Energy (MWh):", pyo.value(model.bess_energy))
    print("Cost:", round(pyo.value(model.cost),0))

    with open('optimization_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Hour', 'Solar (MW)', 'Charge (MW)', 'Discharge (MW)', 'SOC (MWh)', 'Energy Served (MW)'])
        for t in T:
            writer.writerow([
                t,
                solar_profile[t] * pyo.value(model.solar_capacity),
                pyo.value(model.charge[t]),
                pyo.value(model.discharge[t]),
                pyo.value(model.soc[t]),
                pyo.value(model.energy_served_t[t])
            ])

        levcost = 1000 * lcoe(load * 8760 * target,pyo.value(model.cost),0,0.08,20)
        print("lcoe:", round(levcost,1))

    return pyo.value(model.cost), pyo.value(model.solar_capacity), pyo.value(model.bess_power), pyo.value(model.bess_energy), lcoe

if __name__ == "__main__":
    # Setting up environment
    latitude = 19.4326
    longitude = 99.1332
    print("getting solar profile...")
    yearly_profile = generate_hourly_solar_profile(latitude, longitude, year=2023)
    print("got solar profile")
    # demand_profile = np.full(len(yearly_profile), 100)  # Demand profile in MW

    # Run optimization
    optimise_bess(yearly_profile)