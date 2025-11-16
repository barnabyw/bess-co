from Code.archive.assumptions import *
from profile import generate_hourly_solar_profile

import pyomo.environ as pyo
import pandas as pd
import numpy as np
import time
from pyomo.opt import SolverFactory

CWD = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CWD, "..", "inputs")

countries_df = pd.read_csv(os.path.join(INPUT_PATH, "all_country_coordinates_2.csv"))

#===Model Setup===
# -----------------------------

def optimise_bess(
    solar_profile,
    solar_capex,
    bess_energy_capex,
    load=1.0,             # [MW] Average load to serve
    availability=0.8,     # [%] Target availability or percentage of demand to meet
    efficiency=0.9,       # [%] Round-trip efficiency approx
    start_soc=0.5,        # [%] Initial state of charge (fraction of energy capacity)
    return_timeseries=False
):
    """
    Optimizes Solar and BESS capacity to meet a specified demand target at minimum cost.

    Returns:
        tuple: (cost, solar_capacity, bess_energy, results_data)
            - cost (float): total minimized capital cost.
            - solar_capacity (float): optimal solar capacity in MW.
            - bess_energy (float): optimal BESS energy capacity in MWh.
            - results_data (pd.DataFrame or None): hourly results if requested, with:
                Hour
                Solar_Gen_MWh
                Solar_Used_MWh
                Solar_Charge_MWh
                Solar_Curtailed_MWh
                BESS_Discharge_MWh
                SOC_MWh
                Energy_Served_MWh
                Energy_Unserved_MWh
    """
    periods = len(solar_profile)
    demand = np.full(periods, load)
    T = range(periods)

    model = pyo.ConcreteModel(name="Solar_BESS_Optimization")
    model.T = pyo.Set(initialize=T)

    # --- Decision Variables ---
    model.solar_capacity = pyo.Var(within=pyo.NonNegativeReals)
    model.bess_energy    = pyo.Var(within=pyo.NonNegativeReals)

    model.soc            = pyo.Var(model.T, within=pyo.NonNegativeReals)

    model.bess_charge    = pyo.Var(model.T, within=pyo.NonNegativeReals)  # charging power (MWh per step)
    model.bess_discharge = pyo.Var(model.T, within=pyo.NonNegativeReals)  # discharging power

    model.solar_used     = pyo.Var(model.T, within=pyo.NonNegativeReals)  # solar directly to load
    model.curtail        = pyo.Var(model.T, within=pyo.NonNegativeReals)  # curtailed solar

    model.energy_served_t = pyo.Var(model.T, within=pyo.NonNegativeReals) # to load

    # --- Constraints ---

    # SoC balance
    def soc_balance_rule(m, t):
        if t == 0:
            return m.soc[t] == m.bess_energy * start_soc
        # charge adds (with efficiency), discharge removes (with losses)
        return m.soc[t] == (
            m.soc[t-1]
            + m.bess_charge[t] * efficiency
            - m.bess_discharge[t] / efficiency
        )
    model.soc_balance = pyo.Constraint(model.T, rule=soc_balance_rule)

    # Storage capacity limit
    def soc_limit_rule(m, t):
        return m.soc[t] <= m.bess_energy
    model.soc_limit = pyo.Constraint(model.T, rule=soc_limit_rule)

    # Solar generation balance
    def solar_balance_rule(m, t):
        return (
            m.solar_capacity * solar_profile[t]
            == m.solar_used[t] + m.bess_charge[t] + m.curtail[t]
        )
    model.solar_balance = pyo.Constraint(model.T, rule=solar_balance_rule)

    # Load balance (served energy)
    def load_balance_rule(m, t):
        return m.energy_served_t[t] == m.solar_used[t] + m.bess_discharge[t]
    model.load_balance = pyo.Constraint(model.T, rule=load_balance_rule)

    # Demand cap per timestep
    def demand_limit_rule(m, t):
        return m.energy_served_t[t] <= demand[t]
    model.demand_limit = pyo.Constraint(model.T, rule=demand_limit_rule)

    # Power limits for charge and discharge (can change if you want separate inverter sizes)
    model.power_limits = pyo.ConstraintList()
    for t in T:
        model.power_limits.add(model.bess_charge[t]    <= model.solar_capacity)  # max charge power
        model.power_limits.add(model.bess_discharge[t] <= model.solar_capacity)  # max discharge power

    # Availability constraint over the horizon
    model.availability_constraint = pyo.Constraint(
        expr=sum(model.energy_served_t[t] for t in T) >= availability * sum(demand)
    )

    # Objective: minimise CAPEX
    model.cost = pyo.Objective(
        expr=model.solar_capacity * solar_capex +
             model.bess_energy * bess_energy_capex,
        sense=pyo.minimize
    )

    # --- Solve ---
    solver = SolverFactory("cbc")
    solver.solve(model, tee=False)

    cost         = pyo.value(model.cost)
    solar_cap    = pyo.value(model.solar_capacity)
    bess_energy  = pyo.value(model.bess_energy)

    results_data = None
    if return_timeseries:
        solar_gen   = [solar_cap * solar_profile[t]             for t in T]
        solar_used  = [pyo.value(model.solar_used[t])           for t in T]
        solar_chg   = [pyo.value(model.bess_charge[t])          for t in T]
        curtail     = [pyo.value(model.curtail[t])              for t in T]
        bess_dis    = [pyo.value(model.bess_discharge[t])       for t in T]
        soc         = [pyo.value(model.soc[t])                  for t in T]
        served      = [pyo.value(model.energy_served_t[t])      for t in T]
        unserved    = [demand[t] - served[t]                    for t in T]

        results_data = pd.DataFrame({
            "Hour": list(T),
            "Solar_Gen_MWh":        solar_gen,
            "Solar_Used_MWh":       solar_used,
            "Solar_Charge_MWh":     solar_chg,
            "Solar_Curtailed_MWh":  curtail,
            "BESS_Discharge_MWh":   bess_dis,
            "SOC_MWh":              soc,
            "Energy_Served_MWh":    served,
            "Energy_Unserved_MWh":  unserved,
        })

    return cost, solar_cap, bess_energy, results_data

def optimise_availability(
    solar_profile,
    solar_capacity,
    bess_energy,
    load,
    efficiency=0.9,
    start_soc=0.5
):
    """
    Dispatch optimiser for fixed Solar + BESS.
    Matches the new optimise_bess() formulation exactly.
    """

    periods = len(solar_profile)
    T = range(periods)

    demand = np.full(periods, load)

    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=T)

    # --- Variables ---
    model.bess_charge    = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.bess_discharge = pyo.Var(model.T, within=pyo.NonNegativeReals)

    model.solar_used     = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.curtail        = pyo.Var(model.T, within=pyo.NonNegativeReals)

    model.soc            = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.energy_served_t = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # --- Constraints ---

    # SoC balance
    def soc_balance_rule(m, t):
        if t == 0:
            return m.soc[t] == bess_energy * start_soc
        return m.soc[t] == (
            m.soc[t-1]
            + m.bess_charge[t] * efficiency
            - m.bess_discharge[t] / efficiency
        )
    model.soc_balance = pyo.Constraint(model.T, rule=soc_balance_rule)

    # Storage limit
    def soc_limit_rule(m, t):
        return m.soc[t] <= bess_energy
    model.soc_limit = pyo.Constraint(model.T, rule=soc_limit_rule)

    # Solar balance
    def solar_balance_rule(m, t):
        return (
            solar_capacity * solar_profile[t]
            == m.solar_used[t] + m.bess_charge[t] + m.curtail[t]
        )
    model.solar_balance = pyo.Constraint(model.T, rule=solar_balance_rule)

    # Load served
    def load_rule(m, t):
        return m.energy_served_t[t] == m.solar_used[t] + m.bess_discharge[t]
    model.load_balance = pyo.Constraint(model.T, rule=load_rule)

    # Demand limit
    def demand_rule(m, t):
        return m.energy_served_t[t] <= demand[t]
    model.demand_limit = pyo.Constraint(model.T, rule=demand_rule)

    # Charge/discharge limits (same inverter assumption)
    model.power_limits = pyo.ConstraintList()
    for t in T:
        model.power_limits.add(model.bess_charge[t]    <= solar_capacity)
        model.power_limits.add(model.bess_discharge[t] <= solar_capacity)

    # Objective: maximise availability (total served)
    model.obj = pyo.Objective(
        expr=sum(model.energy_served_t[t] for t in T),
        sense=pyo.maximize
    )

    # --- Solve ---
    solver = pyo.SolverFactory("cbc")
    result = solver.solve(model, tee=False)

    if result.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print("⚠️ Infeasible — returning availability = 0")
        return 0.0, pd.DataFrame()

    # --- Extract results ---
    served   = [pyo.value(model.energy_served_t[t]) for t in T]
    total_energy = sum(served)
    total_demand = sum(demand)
    availability = total_energy / total_demand

    df = pd.DataFrame({
        "Hour": list(T),
        "Solar_Gen_MWh":        [solar_capacity * solar_profile[t] for t in T],
        "Solar_Used_MWh":       [pyo.value(model.solar_used[t]) for t in T],
        "Solar_Charge_MWh":     [pyo.value(model.bess_charge[t]) for t in T],
        "Solar_Curtailed_MWh":  [pyo.value(model.curtail[t]) for t in T],
        "BESS_Discharge_MWh":   [pyo.value(model.bess_discharge[t]) for t in T],
        "SOC_MWh":              [pyo.value(model.soc[t]) for t in T],
        "Energy_Served_MWh":    served,
        "Energy_Unserved_MWh":  [demand[t] - served[t] for t in T],
    })

    return availability, df


if __name__ == "__main__":
    latitude = 19.4326
    longitude = 99.1332
    #country = "Australia"
    """
    # Filter the row where 'Country' matches
    row = countries_df[countries_df['Country'] == country]

    if not row.empty:
        latitude = row.iloc[0]['Latitude']
        longitude = row.iloc[0]['Longitude']
        print(f"{country}: lat={latitude}, lon={longitude}")
    else:
        print(f"Country '{country}' not found in DataFrame.")
    """
    solar_profile = generate_hourly_solar_profile(latitude, longitude, solar_year=2023)
    print("got solar profile")
    profile = solar_profile
    solar_capex = 690
    bess_energy_capex = 191
    cost, solar_capacity, bess_energy, results_1 = optimise_bess(solar_profile, solar_capex, bess_energy_capex)
    print(f"solar cap is {solar_capacity}, bess is {bess_energy}")
    availability, results_2 = optimise_availability(profile, solar_capacity, bess_energy, 1)
    results_1.to_csv(r'C:\Users\barna\OneDrive\Documents\Solar_BESS results\opti_results.csv')
    results_2.to_csv(r'C:\Users\barna\OneDrive\Documents\Solar_BESS results\avail_results.csv')
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
    
    
    wind test
        #latitude = 55.3781
    #longitude = 3.4360
    wind_profile = parse_renewables_ninja(r"C:\\Users\\barna\Downloads\\ninja_wind_54.7867_-1.9809_corrected.csv")
"""