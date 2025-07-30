import pyomo.environ as pyo
import numpy as np
import csv

# Time horizon (e.g., 24 hours)
T = range(72)
demand = np.full(72, 100)  # 100 MW constant demand

model = pyo.ConcreteModel()

# Sets
model.T = pyo.Set(initialize=T)

# Parameters
solar_profile = np.random.rand(24)
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

# normalized solar generation profile
solar_cost_per_mw = 1000
bess_power_cost_per_mw = 500
bess_energy_cost_per_mwh = 200
efficiency = 0.9

# Decision Variables
model.solar_capacity = pyo.Var(within=pyo.NonNegativeReals)
model.bess_power = pyo.Var(within=pyo.NonNegativeReals)
model.bess_energy = pyo.Var(within=pyo.NonNegativeReals)
model.charge = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.discharge = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.soc = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.energy_served_t = pyo.Var(model.T, within=pyo.NonNegativeReals)

# Constraints
def soc_balance_rule(m, t):
    if t == 0:
        return m.soc[t] == m.charge[t] * efficiency - m.discharge[t] / efficiency
    return m.soc[t] == m.soc[t-1] + m.charge[t] * efficiency - m.discharge[t] / efficiency
model.soc_balance = pyo.Constraint(model.T, rule=soc_balance_rule)

def energy_served_t_rule(m, t):
    return m.energy_served_t[t] == m.solar_capacity * solar_profile[t] + m.discharge[t] - m.charge[t]

model.energy_served_t_constraint = pyo.Constraint(model.T, rule=energy_served_t_rule)

model.soc_limit = pyo.ConstraintList()
for t in T:
    model.soc_limit.add(model.soc[t] <= model.bess_energy)
    model.soc_limit.add(model.charge[t] <= model.bess_power)
    model.soc_limit.add(model.discharge[t] <= model.bess_power)

# Energy served constraints
def energy_served_rule(m):
    return sum(m.energy_served_t[t] for t in m.T) >= 0.95 * sum(demand)
model.energy_served = pyo.Constraint(rule=energy_served_rule)

model.energy_served_limit = pyo.ConstraintList()
for t in T:
    model.energy_served_limit.add(model.energy_served_t[t] <= demand[t])
    model.energy_served_limit.add(model.discharge[t] <= model.soc[t]) #added
    model.energy_served_limit.add(model.charge[t] <= model.solar_capacity * solar_profile[t])

# Objective: Minimize total cost
model.cost = pyo.Objective(
    expr=model.solar_capacity * solar_cost_per_mw +
         model.bess_power * bess_power_cost_per_mw +
         model.bess_energy * bess_energy_cost_per_mwh,
    sense=pyo.minimize
)

print("Optimising...")
# Solver
solver = pyo.SolverFactory('glpk')
solver.solve(model)

# Results
print("Optimal Solar Capacity (MW):", pyo.value(model.solar_capacity))
print("Optimal BESS Power (MW):", pyo.value(model.bess_power))
print("Optimal BESS Energy (MWh):", pyo.value(model.bess_energy))

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

