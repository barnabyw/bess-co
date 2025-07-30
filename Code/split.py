import time

import pyomo.environ as pyo
import numpy as np
import csv

from assumptions import solar_cost_per_mw, load, bess_power_cost_per_mw, bess_energy_cost_per_mwh, efficiency, M, start_soc
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

penalty_weight = 1e-3

def optimise_bess(solar_profile):
    periods = len(solar_profile)
    demand = np.full(periods, load)
    T_full = range(periods)

    segment_count = 2
    segment_length = periods // segment_count

    model = pyo.ConcreteModel()

    # Shared sizing variables
    model.solar_capacity = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_power = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_energy = pyo.Var(within=pyo.NonNegativeReals)

    # Blocks for each segment
    model.segments = pyo.Block(range(segment_count))

    soc_end_prev = None  # For SoC continuity

    all_energy_served = []

    for seg in range(segment_count):
        block = model.segments[seg]
        start_t = seg * segment_length
        end_t = (seg + 1) * segment_length if seg < segment_count - 1 else periods
        block.T = range(start_t, end_t)

        # Segment-local decision variables
        block.charge = pyo.Var(block.T, within=pyo.NonNegativeReals)
        block.discharge = pyo.Var(block.T, within=pyo.NonNegativeReals)
        block.soc = pyo.Var(block.T, within=pyo.NonNegativeReals)
        block.energy_served_t = pyo.Var(block.T, within=pyo.NonNegativeReals)
        block.penalty = pyo.Var(block.T, within=pyo.NonNegativeReals)

        # SoC balance
        def soc_balance_rule(m, t):
            if t == start_t:
                return m.soc[t] == (
                    model.bess_energy * start_soc if soc_end_prev is None else soc_end_prev
                )
            return m.soc[t] == m.soc[t - 1] + m.charge[t] * efficiency - m.discharge[t] / efficiency

        block.soc_balance = pyo.Constraint(block.T, rule=soc_balance_rule)

        # Energy served rule
        def energy_served_rule(m, t):
            return m.energy_served_t[t] == model.solar_capacity * solar_profile[t] + m.discharge[t] - m.charge[t]

        block.energy_served_t_constraint = pyo.Constraint(block.T, rule=energy_served_rule)

        # Limits
        block.bess_limits = pyo.ConstraintList()
        for t in block.T:
            block.bess_limits.add(block.soc[t] <= model.bess_energy)
            block.bess_limits.add(block.charge[t] <= model.bess_power)
            block.bess_limits.add(block.discharge[t] <= model.bess_power)
            block.bess_limits.add(block.discharge[t] <= block.soc[t])
            block.bess_limits.add(block.charge[t] <= model.solar_capacity * solar_profile[t])
            block.bess_limits.add(block.penalty[t] >= block.charge[t] + block.discharge[t] - model.bess_power)

        # Demand cap
        block.energy_served_limit = pyo.ConstraintList()
        for t in block.T:
            block.energy_served_limit.add(block.energy_served_t[t] <= demand[t])

        # Save SoC at end of block for next
        soc_end_prev = block.soc[end_t - 1]

        # Collect energy served for overall constraint
        all_energy_served += [block.energy_served_t[t] for t in block.T]

    # Availability constraint (global)
    model.energy_served_total = pyo.Constraint(
        expr=sum(all_energy_served) >= 0.95 * sum(demand)
    )

    # Objective
    model.cost = pyo.Objective(
        expr=model.solar_capacity * solar_cost_per_mw +
             model.bess_power * bess_power_cost_per_mw +
             model.bess_energy * bess_energy_cost_per_mwh +
             penalty_weight * sum(
            model.segments[seg].penalty[t] for seg in range(segment_count) for t in model.segments[seg].T),
        sense=pyo.minimize
    )

    # Solve
    print("Optimising...")
    start_time = time.time()
    solver = pyo.SolverFactory('cbc')
    solver.solve(model)
    end_time = time.time()
    print(f"Optimisation completed in {round(end_time - start_time, 1)} seconds")

    # Output
    print("Optimal Solar Capacity (MW):", pyo.value(model.solar_capacity))
    print("Optimal BESS Power (MW):", pyo.value(model.bess_power))
    print("Optimal BESS Energy (MWh):", pyo.value(model.bess_energy))
    print("Cost:", round(pyo.value(model.cost), 0))

    # Save results
    with open('optimization_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Hour', 'Solar (MW)', 'Charge (MW)', 'Discharge (MW)', 'SOC (MWh)', 'Energy Served (MW)'])
        for seg in range(segment_count):
            blk = model.segments[seg]
            for t in blk.T:
                writer.writerow([
                    t,
                    solar_profile[t] * pyo.value(model.solar_capacity),
                    pyo.value(blk.charge[t]),
                    pyo.value(blk.discharge[t]),
                    pyo.value(blk.soc[t]),
                    pyo.value(blk.energy_served_t[t])
                ])

    return pyo.value(model.cost), pyo.value(model.solar_capacity), pyo.value(model.bess_power), pyo.value(
        model.bess_energy)

if __name__ == "__main__":
    latitude = 23.8634
    longitude = 69.1328
    print("getting solar profile...")
    yearly_profile = generate_hourly_solar_profile(latitude, longitude, year=2023)
    print("got solar profile")
    # demand_profile = np.full(len(yearly_profile), 100)  # Demand profile in MW

    # Run optimization
    optimise_bess(yearly_profile)